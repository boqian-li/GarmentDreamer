import os
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import nvdiffrast.torch as dr

from kiui.mesh import Mesh
from kiui.mesh_utils import clean_mesh, decimate_mesh
from kiui.op import safe_normalize, scale_img_hwc, make_divisible, uv_padding
from kiui.cam import orbit_camera, get_perspective, look_at
from netf.render.texture_encoder import MLP, HashGridEncoder, FrequencyEncoder, TriplaneEncoder
import copy

def render_mesh(
        glctx, 
        v, f, 
        vt, ft, albedo, vc,
        vn, fn,
        pose, proj, 
        h0, w0, 
        ssaa=1, bg_color=1, 
        texture_filter='linear-mipmap-linear', 
        color_activation=None,
    ):
    
    # do super-sampling
    if ssaa != 1:
        h = make_divisible(h0 * ssaa, 8)
        w = make_divisible(w0 * ssaa, 8)
    else:
        h, w = h0, w0
    
    results = {}

    pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
    proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

    # get v_clip and render rgb
    v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
    v_clip = v_cam @ proj.T

    rast, rast_db = dr.rasterize(glctx, v_clip, f, (h, w))

    alpha = (rast[0, ..., 3:] > 0).float()
    depth, _ = dr.interpolate(-v_cam[..., [2]], rast, f) # [1, H, W, 1]
    depth = depth.squeeze(0) # [H, W, 1]

    if vc is not None:
        # use vertex color
        color, _ = dr.interpolate(vc.unsqueeze(0).contiguous(), rast, f)
    else:
        # use texture image
        texc, texc_db = dr.interpolate(vt.unsqueeze(0).contiguous(), rast, ft, rast_db=rast_db, diff_attrs='all')
        color = dr.texture(albedo.unsqueeze(0), texc, uv_da=texc_db, filter_mode=texture_filter) # [1, H, W, 3]

    if color_activation is not None:
        color = color_activation(color)

    # antialias
    color = dr.antialias(color, rast, v_clip, f).squeeze(0) # [H, W, 3]
    color = alpha * color + (1 - alpha) * bg_color

    # get vn and render normal
    if vn is None:
        i0, i1, i2 = f[:, 0].long(), f[:, 1].long(), f[:, 2].long()
        v0, v1, v2 = v[i0, :], v[i1, :], v[i2, :]

        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = safe_normalize(face_normals)
        
        vn = torch.zeros_like(v)
        vn.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

        vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))
    
    normal, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, fn)
    normal = safe_normalize(normal[0])

    # rotated normal (where [0, 0, 1] always faces camera)
    rot_normal = normal @ pose[:3, :3]
    viewcos = rot_normal[..., [2]]

    # ssaa
    if ssaa != 1:
        color = scale_img_hwc(color, (h0, w0))
        alpha = scale_img_hwc(alpha, (h0, w0))
        depth = scale_img_hwc(depth, (h0, w0))
        normal = scale_img_hwc(normal, (h0, w0))
        viewcos = scale_img_hwc(viewcos, (h0, w0))

    results['image'] = color.clamp(0, 1)
    results['alpha'] = alpha
    results['depth'] = depth
    results['normal'] = (normal + 1) / 2
    results['viewcos'] = viewcos

    return results

class Renderer(nn.Module):
    def __init__(self, args, view_sampler, device):
        
        super().__init__()

        self.args = args
        self.device = device
        self.view_sampler = view_sampler

        self.mesh = Mesh.load(self.args.mesh, resize=False, bound=1.0, front_dir=self.args.front_dir)

        # it's necessary to clean the mesh to facilitate later remeshing!
        vertices = self.mesh.v.detach().cpu().numpy()
        triangles = self.mesh.f.detach().cpu().numpy()
        vertices, triangles = clean_mesh(vertices, triangles, min_f=32, min_d=10, remesh=False)
        self.mesh.v = torch.from_numpy(vertices).contiguous().float().to(self.device)
        self.mesh.f = torch.from_numpy(triangles).contiguous().int().to(self.device)

        if not self.args.force_cuda_rast:
            self.glctx = dr.RasterizeGLContext()
        else:
            self.glctx = dr.RasterizeCudaContext()
        
        # extract trainable parameters
        self.v_offsets = nn.Parameter(torch.zeros_like(self.mesh.v))
        
        # texture
        if self.args.tex_mode == 'hashgrid':
            self.encoder = HashGridEncoder().to(self.device)
        elif self.args.tex_mode == 'mlp':
            self.encoder = FrequencyEncoder().to(self.device)
        elif self.args.tex_mode == 'triplane':
            self.encoder = TriplaneEncoder().to(self.device)
        else:
            raise NotImplementedError(f"unsupported texture mode: {self.args.tex_mode} for {self.args.geom_mode}")
        
        print(self.encoder.output_dim)
        self.mlp = MLP(self.encoder.output_dim, 3, 32, 2, bias=True).to(self.device)  # TODO

        # init hashgrid texture from mesh
        if self.args.fit_tex:
            self.fit_texture_from_mesh(self.args.fit_tex_iters, os.path.join(self.args.outdir, 'final_mesh' + '_fitted_texture.obj'))

    def render_mesh(self, pose, proj, h, w, ssaa=1, bg_color=1):
        return render_mesh(
            self.glctx, 
            self.mesh.v, self.mesh.f, self.mesh.vt, 
            self.mesh.ft, self.mesh.albedo, 
            self.mesh.vc, self.mesh.vn, self.mesh.fn, 
            pose, proj, h, w, 
            ssaa=ssaa, bg_color=bg_color,
        )
    
    def fit_texture_from_mesh(self, iters=512, save_path=None):
        # a small training loop...

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters(), 'lr': self.args.hashgrid_lr},
            {'params': self.mlp.parameters(), 'lr': self.args.mlp_lr},
        ])

        resolution = 1024

        # print(f"[INFO] fitting texture...")
        pbar = tqdm.trange(iters)
        for i in pbar:
            views_subset = self.view_sampler()
            assert len(views_subset) == 1
            view = views_subset[0]
            
            position_ = view.camera.C2W[:3, 3]
            position = np.zeros_like(position_)

            position[0] = position_[1]
            position[1] = position_[2]
            position[2] = position_[0]

            rotation = look_at(campos=position, target= np.zeros([3], dtype=np.float32))

            position[0] = - position[0]
            rotation[0, 1] = - rotation[0, 1]
            rotation[0, 2] = - rotation[0, 2]
            rotation[:, 0] = np.cross(rotation[:, 2], rotation[:, 1])
            rotation[:, 0] = rotation[:, 0] / np.linalg.norm(rotation[:, 0])

            position[2] = - position[2]
            rotation[2, 1] = - rotation[2, 1]
            rotation[2, 2] = - rotation[2, 2]
            rotation[:, 0] = np.cross(rotation[:, 2], rotation[:, 1])
            rotation[:, 0] = rotation[:, 0] / np.linalg.norm(rotation[:, 0])

            position[0] = - position[0]
            rotation[0, 1] = - rotation[0, 1]
            rotation[0, 2] = - rotation[0, 2]
            rotation[:, 0] = np.cross(rotation[:, 2], rotation[:, 1])
            rotation[:, 0] = rotation[:, 0] / np.linalg.norm(rotation[:, 0])

            position[2] = - position[2]
            rotation[2, 1] = - rotation[2, 1]
            rotation[2, 2] = - rotation[2, 2]
            rotation[:, 0] = np.cross(rotation[:, 2], rotation[:, 1])
            rotation[:, 0] = rotation[:, 0] / np.linalg.norm(rotation[:, 0])


            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = position
            # pose[1, :] *= -1

            # proj = get_perspective(self.args.fovy)
            proj = self.projection(fx=view.camera.K[0,0],
                                                fy=view.camera.K[1,1],
                                                cx=view.camera.K[0,2],
                                                cy=view.camera.K[1,2],
                                                width=int(view.rgb.shape[0]),
                                                height=int(view.rgb.shape[1]))

            
            pred = self.render(pose, proj, resolution, resolution)
            image_pred = pred['image']
            normal_pred = pred['normal']

            mask = (pred['alpha'].squeeze() > 0) & (torch.flipud(view.mask.squeeze()) > 0) & (pred['cosinesview'] <= 0) # TODO: view direction mask!

            loss = loss_fn(image_pred[mask], torch.flipud(view.rgb)[mask]) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"MSE = {loss.item():.6f}")
        
        # print(f"[INFO] finished fitting texture!")
        self.export_mesh(save_path, texture_resolution=self.args.texture_resolution, reverse=True)
        print(f"save final mesh with reconstructed texture to {save_path}.")

    def projection(self, fx, fy, cx, cy, width, height, n=0.01, f=1000):
        return np.array([[2.0*fx/width,           0,       1.0 - 2.0 * cx / width,                  0],
                            [         0, 2.0*fy/height,      1.0 - 2.0 * cy / height,                  0],
                            [         0,             0,                 -(f+n)/(f-n),     -(2*f*n)/(f-n)],
                            [         0,             0,                           -1,                  0.0]], dtype=np.float32) 

    def get_params(self):

        params = [
            {'params': self.encoder.parameters(), 'lr': self.args.hashgrid_lr},
            {'params': self.mlp.parameters(), 'lr': self.args.mlp_lr},
        ]

        if not self.args.fix_geo:
            params.append({'params': self.v_offsets, 'lr': self.args.geom_lr})

        return params

    @torch.no_grad()
    def export_mesh(self, save_path, texture_resolution=2048, padding=16, reverse=False):

        mesh = Mesh(v=self.v, f=self.f, albedo=None, device=self.device)

        # print(f"[INFO] uv unwrapping...")
        mesh.auto_normal()
        mesh.auto_uv(vmap=False)

        # render uv maps
        h = w = texture_resolution
        uv = mesh.vt * 2.0 - 1.0 # uvs to range [-1, 1]
        uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

        rast, _ = dr.rasterize(self.glctx, uv.unsqueeze(0), mesh.ft, (h, w)) # [1, h, w, 4]

        # masked query 
        xyzs, _ = dr.interpolate(mesh.v.unsqueeze(0), rast, mesh.f) # [1, h, w, 3]
        mask, _ = dr.interpolate(torch.ones_like(mesh.v[:, :1]).unsqueeze(0), rast, mesh.f) # [1, h, w, 1]
        xyzs = xyzs.view(-1, 3)
        mask = (mask > 0).view(-1)
        
        albedo = torch.zeros(h * w, 3, device=self.device, dtype=torch.float32)

        if mask.any():
            # print(f"[INFO] querying texture...")

            xyzs = xyzs[mask] # [M, 3]

            # batched inference to avoid OOM
            batch = []
            head = 0
            while head < xyzs.shape[0]:
                tail = min(head + 640000, xyzs.shape[0])
                batch.append(torch.sigmoid(self.mlp(self.encoder(xyzs[head:tail]))).float())
                head += 640000

            albedo[mask] = torch.cat(batch, dim=0)
        
        albedo = albedo.view(h, w, -1)
        mask = mask.view(h, w)

        # print(f"[INFO] uv padding...")
        albedo = uv_padding(albedo, mask, padding)

        if reverse:
            # Flip mesh vertices
            new_mesh = copy.deepcopy(mesh)
            new_mesh.v[:, 0] = -new_mesh.v[:, 0]
            new_mesh.albedo = albedo
            new_mesh.write(save_path)
        else:
            mesh.albedo = albedo
            mesh.write(save_path)

    @property
    def v(self):
        if self.args.fix_geo:
            return self.mesh.v
        else:
            return self.mesh.v + self.v_offsets
    
    @property
    def f(self):
        return self.mesh.f

    @torch.no_grad()
    def remesh(self):
        vertices = self.v.detach().cpu().numpy()
        triangles = self.f.detach().cpu().numpy()
        vertices, triangles = clean_mesh(vertices, triangles, repair=False, remesh=True, remesh_size=self.args.remesh_size)
        if self.args.decimate_target > 0 and triangles.shape[0] > self.args.decimate_target:
            vertices, triangles = decimate_mesh(vertices, triangles, self.args.decimate_target, optimalplacement=False)
        self.mesh.v = torch.from_numpy(vertices).contiguous().float().to(self.device)
        self.mesh.f = torch.from_numpy(triangles).contiguous().int().to(self.device)
        self.v_offsets = nn.Parameter(torch.zeros_like(self.mesh.v)).to(self.device)
        
    
    def render(self, pose, proj, h0, w0, ssaa=1, bg_color=1):

        # do super-sampling
        if ssaa != 1:
            h = make_divisible(h0 * ssaa, 8)
            w = make_divisible(w0 * ssaa, 8)
        else:
            h, w = h0, w0
        
        results = {}

        # get v
        v = self.v
        f = self.f

        pose = torch.from_numpy(pose.astype(np.float32)).to(v.device)
        proj = torch.from_numpy(proj.astype(np.float32)).to(v.device)

        # get v_clip and render rgb
        v_cam = torch.matmul(F.pad(v, pad=(0, 1), mode='constant', value=1.0), torch.inverse(pose).T).float().unsqueeze(0)
        v_clip = v_cam @ proj.T

        rast, rast_db = dr.rasterize(self.glctx, v_clip, f, (h, w))

        alpha = torch.clamp(rast[..., -1:], 0, 1).contiguous() # [1, H, W, 1]
        alpha = dr.antialias(alpha, rast, v_clip, f).clamp(0, 1).squeeze(0) # important to enable gradients!

        depth, _ = dr.interpolate(-v_cam[..., [2]], rast, f) # [1, H, W, 1]
        depth = depth.squeeze(0) # [H, W, 1]

        xyzs_, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, H, W, 3]
        xyzs = xyzs_.view(-1, 3)
        mask = (alpha > 0).view(-1)
        color = torch.zeros_like(xyzs, dtype=torch.float32)
        if mask.any():
            masked_albedo = torch.sigmoid(self.mlp(self.encoder(xyzs[mask], bound=1)))
            color[mask] = masked_albedo.float()
        color = color.view(1, h, w, 3)

        # antialias
        color = dr.antialias(color, rast, v_clip, f).clamp(0, 1).squeeze(0) # [H, W, 3]
        color = alpha * color + (1 - alpha) * bg_color

        # get vn and render normal
        if self.args.fix_geo:
            vn = self.mesh.vn
        else:
            i0, i1, i2 = f[:, 0].long(), f[:, 1].long(), f[:, 2].long()
            v0, v1, v2 = v[i0, :], v[i1, :], v[i2, :]

            face_normals = torch.cross(v1 - v0, v2 - v0)
            face_normals = safe_normalize(face_normals)

            vn = torch.zeros_like(v)
            vn.scatter_add_(0, i0[:, None].repeat(1,3), face_normals)
            vn.scatter_add_(0, i1[:, None].repeat(1,3), face_normals)
            vn.scatter_add_(0, i2[:, None].repeat(1,3), face_normals)

            vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=vn.device))

        normal_, _ = dr.interpolate(vn.unsqueeze(0).contiguous(), rast, f)
        normal = safe_normalize(normal_[0])


        # compute view direction mask for only outside of surface
        with torch.no_grad():
            position = dr.antialias(xyzs_, rast, v_clip, f)[0]
            normal_ = dr.antialias(normal_, rast, v_clip, f)[0]
            view_direction = F.normalize(position - pose[:3, 3], dim=-1)
            cosines_view = F.cosine_similarity(view_direction, normal_, dim=-1, eps=1e-6)

        # rotated normal (where [0, 0, 1] always faces camera)
        # rot_normal = normal @ pose[:3, :3]
        # viewcos = rot_normal[..., [2]]

        # ssaa
        if ssaa != 1:
            color = scale_img_hwc(color, (h0, w0))
            alpha = scale_img_hwc(alpha, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            normal = scale_img_hwc(normal, (h0, w0))
            # viewcos = scale_img_hwc(viewcos, (h0, w0))

        results['image'] = color
        results['alpha'] = alpha
        results['depth'] = depth
        results['normal'] = (normal + 1) / 2
        results['cosinesview'] = cosines_view
        # results['viewcos'] = viewcos

        return results