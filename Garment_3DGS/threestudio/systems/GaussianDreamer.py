from dataclasses import dataclass, field
import torch
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussiansplatting.scene.cameras import Camera
from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import SH2RGB
from gaussiansplatting.scene.gaussian_model import BasicPointCloud
import open3d as o3d
import numpy as np
import io  
from PIL import Image  

from gaussiansplatting.utils.graphics_utils import fov2focal, focal2fov
import json


def load_ply(path,save_path):
    C0 = 0.28209479177387814
    def SH2RGB(sh):
        return sh * C0 + 0.5
    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    color = SH2RGB(features_dc[:,:,0])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(save_path, point_cloud)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


@threestudio.register("gaussiandreamer-system")
class GaussianDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        sh_degree: int = 0
        scale: float = 0.4
        alpha_threshold: float = 0.88
        deviation: float = 0.01
        num_pts_space: int = 500000
        load_path: str = "./load/shapes/stand.obj"



    cfg: Config
    def configure(self) -> None:
        self.radius = self.cfg.radius
        self.scale = self.cfg.scale
        self.sh_degree =self.cfg.sh_degree
        self.load_path = self.cfg.load_path

        self.gaussian = GaussianModel(sh_degree = self.sh_degree)
        self.alpha_threshold = self.cfg.alpha_threshold

        self.num_pts_space = self.cfg.num_pts_space
        self.deviation = self.cfg.deviation
        # bg_color = [1, 1, 1] if False else [0, 0, 0]
        bg_color = [1, 1, 1]
        self.background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    
    def save_gif_to_file(self,images, output_file):  
        with io.BytesIO() as writer:  
            images[0].save(  
                writer, format="GIF", save_all=True, append_images=images[1:], duration=100, loop=0  
            )  
            writer.seek(0)  
            with open(output_file, 'wb') as file:  
                file.write(writer.read())
    

    
    def add_points(self,coords,rgb):
        pcd_by3d = o3d.geometry.PointCloud()
        pcd_by3d.points = o3d.utility.Vector3dVector(np.array(coords))
        

        bbox = pcd_by3d.get_axis_aligned_bounding_box()
        np.random.seed(0)

        num_points = self.num_pts_space  
        points = np.random.uniform(low=np.asarray(bbox.min_bound), high=np.asarray(bbox.max_bound), size=(num_points, 3))


        kdtree = o3d.geometry.KDTreeFlann(pcd_by3d)


        points_inside = []
        color_inside= []
        print(f"deviation from template mesh: {self.deviation}")
        for point in points:
            _, idx, _ = kdtree.search_knn_vector_3d(point, 1)
            nearest_point = np.asarray(pcd_by3d.points)[idx[0]]
            if np.linalg.norm(point - nearest_point) < self.deviation:  # 这个阈值可能需要调整
                points_inside.append(point)
                color_inside.append(rgb[idx[0]]+0.2*np.random.random(3))

        all_coords = np.array(points_inside)
        all_rgb = np.array(color_inside)
        all_coords = np.concatenate([all_coords,coords],axis=0)
        all_rgb = np.concatenate([all_rgb,rgb],axis=0)
        return all_coords,all_rgb


    def template(self):
        self.num_pts  = 50000
        mesh = o3d.io.read_triangle_mesh(self.load_path)
        point_cloud = mesh.sample_points_uniformly(number_of_points=self.num_pts)
        coords = np.array(point_cloud.points)

        coords = coords 

        shs = np.random.random((self.num_pts, 3)) / 255.0
        rgb = SH2RGB(shs)
        adjusment = np.zeros_like(coords)
        adjusment[:,0] = coords[:,2]
        adjusment[:,1] = coords[:,0]
        adjusment[:,2] = coords[:,1]

        return adjusment,rgb
    
    # 0.33 blue denium jumpsuit with sleeves, others0.3
    def pcb(self):
        # Since this data set has no colmap data, we start with random points

        coords,rgb = self.template()

        bound= self.radius*self.scale

        all_coords,all_rgb = self.add_points(coords,rgb)
        

        pcd = BasicPointCloud(points=all_coords *bound, colors=all_rgb, normals=np.zeros((all_coords.shape[0], 3)))

        return pcd
    
    
    def forward(self, batch: Dict[str, Any],renderbackground = None) -> Dict[str, Any]:

        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        alphas = []
        self.viewspace_point_list = []

        for id in range(batch['c2w_3dgs'].shape[0]):
            viewpoint_cam  = Camera(c2w = batch['c2w_3dgs'][id],FoVy = batch['fovy'][id],height = batch['height'],width = batch['width'])
            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, _, radii, alpha, depth= render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['alpha'], render_pkg["depth_3dgs"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii,self.radii)
                

            depth =  depth.permute(1, 2, 0)
            image =  image.permute(1, 2, 0)
            alpha = alpha.permute(1, 2, 0)

            images.append(image)
            depths.append(depth)
            alphas.append(alpha)

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        alphas = torch.stack(alphas, 0)
        self.visibility_filter = self.radii>0.0
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        render_pkg['alphas'] = alphas
        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
    
    def training_step(self, batch, batch_idx):

        self.gaussian.update_learning_rate(self.true_global_step)
        
        if self.true_global_step > 500:
            self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.55)

        self.gaussian.update_learning_rate(self.true_global_step)

        out = self(batch) 

        prompt_utils = self.prompt_processor()
        images = out["comp_rgb"]
        
        guidance_out = self.guidance(
            images, prompt_utils, **batch, rgb_as_latents=False,guidance_eval=False
        )
        

        loss = 0.0

        loss = loss + guidance_out['loss_sds'] *self.C(self.cfg.loss['lambda_sds'])

        
        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}



    def on_before_optimizer_step(self, optimizer):

        with torch.no_grad():
            
            if self.true_global_step < 900: # 15000
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step > 300 and self.true_global_step % 100 == 0: # 500 100
                    size_threshold = 20 if self.true_global_step > 500 else None # 3000
                    self.gaussian.densify_and_prune(0.0002 , 0.05, self.cameras_extent, size_threshold) 






    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"gs_check/iter_{self.true_global_step}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass
    
    def save_cameras_json(self, camera_info_list):
        with open(os.path.join(self.get_save_dir(), "cameras.json"), "w") as json_file:
            json.dump(camera_info_list, json_file)

    def test_step(self, batch, batch_idx):
        # only forward one image per step!

        # generate mask (alpha channel)
        # bg_color = [1, 1, 1] if False else [0, 0, 0]
        bg_color = [1, 1, 1]
        alpha_threshold = self.alpha_threshold

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = self(batch,testbackground_tensor)
        alpha_tensor = out['alphas'].squeeze() # (1024, 1024) range(0, 1)
        depth_tensor = out['depth'].squeeze() # (1024, 1024)
        rgb_tensor = out['comp_rgb'].squeeze() # (1024, 1024, 3)
        
        alpha_tensor = alpha_tensor >= alpha_threshold # ensure alpha tensor contains only 2 numbers: 0 and 1. 
        mask_tensor = alpha_tensor


        # append camera info
        id = 0
        Rt = batch["c2w"][id].detach().cpu().numpy()
        C2W = Rt
        pos = C2W[:3, 3]
        rot = C2W[:3, :3]
        rot[:, :] *= -1
        serializable_array_2d = [x.tolist() for x in rot]
        camera_info_ = {"id": batch["index"][id].item(), "img_name": str(batch["index"][id].item()), "width": batch['width'], "height": batch['height'], 
                        "position": pos.tolist(), "rotation": serializable_array_2d, 
                        "fy": fov2focal(batch['fovy'][id], batch['height']), "fx": fov2focal(focal2fov(fov2focal(batch['fovy'][id], batch['height']), batch['width']), batch['width'])}
        self.camera_info_list += [camera_info_]
        
        # save rgb images
        # self.save_image_grid(
        #     f"gs_rendered_rgb/{batch['index'][0]}.png",
        #     (
        #         [
        #             {
        #                 "type": "rgb",
        #                 "img": batch["rgb"][0],
        #                 "kwargs": {"data_format": "HWC"},
        #             }
        #         ]
        #         if "rgb" in batch
        #         else []
        #     )
        #     + [
        #         {
        #             "type": "rgb",
        #             "img": out["comp_rgb"][0],
        #             "kwargs": {"data_format": "HWC"},
        #         },
        #     ]
        #     + (
        #         [
        #             {
        #                 "type": "rgb",
        #                 "img": out["comp_normal"][0],
        #                 "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
        #             }
        #         ]
        #         if "comp_normal" in out
        #         else []
        #     ),
        #     name="test_step",
        #     step=self.true_global_step,
        # )

        # save rgba images
        self.save_image_rgba(
            f"gs_rendered_rgba/{batch['index'][0]}.png",
            rgb_tensor,
            mask_tensor,
            name="test_step",
            step=self.true_global_step,
        )


    def on_test_epoch_end(self):
        save_path = self.get_save_path(f"last_3dgs.ply")
        self.gaussian.save_ply(save_path)
        load_ply(save_path,self.get_save_path(f"last_pointcloud_with_color.ply"))

        self.save_cameras_json(self.camera_info_list)
        


    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")
        
        opt = OptimizationParams(self.parser)
        point_cloud = self.pcb()
        self.cameras_extent = 4.0
        self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)

        self.pipe = PipelineParams(self.parser)
        self.gaussian.training_setup(opt)
        self.camera_info_list = []
        
        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret
