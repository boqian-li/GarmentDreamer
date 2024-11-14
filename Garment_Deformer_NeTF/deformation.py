import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
from argparse import Namespace
import numpy as np
from pathlib import Path
import gpytoolbox
import torch
from torch import Tensor
import torch.nn as nn
from tqdm import tqdm
from deformer.tools import adjust_and_scale_obj
import cv2
import yaml


from deformer.core import (
    Mesh, Renderer
)
from deformer.losses import (
    mask_loss, normal_consistency_loss, laplacian_loss, normal_map_loss, normal_map_loss_enhanced, hole_mask_loss, shading_loss
)
from deformer.modules import (
    SpaceNormalization, NeuralShader, ViewSampler
)
from deformer.utils import (
    AABB, read_views, read_mesh, write_mesh
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def load_config(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Namespace(**config_dict)


def deformation(args):
    
    args_deformation = load_config(args.deformer_config)

    args_deformation.input_dir = args.sample_path
    args_deformation.output_dir = os.path.join(args.sample_path, 'deformation_check')
    args_deformation.initial_mesh = args.template_path
    args_deformation.bound = args.bound

    args_deformation.device = 0 # the first visible gpu!

    args = args_deformation

    # set seed:
    np.random.seed(12)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)


    # Select the device
    device = torch.device('cpu')
    if torch.cuda.is_available() and args.device >= 0:
        device = torch.device(f'cuda:{args.device}')
    
    print(f"Using device {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Create directories
    experiment_dir = Path(args.output_dir)

    # shaders_save_path = experiment_dir / "shaders"
    images_save_path = experiment_dir / "images"
    meshes_save_path = experiment_dir / "meshes"
    images_save_path.mkdir(parents=True, exist_ok=True)
    meshes_save_path.mkdir(parents=True, exist_ok=True)
    # shaders_save_path.mkdir(parents=True, exist_ok=True)

    # Save args for this execution
    with open(experiment_dir / "args.txt", "w") as text_file:
        print(f"{args}", file=text_file)

    # Read the views
    # mode == 0: load txts under views folder
    # mode == 1: load json under dataset folder
    # TODO: read RGB images and store in "views"
    mode = 1
    views = read_views(args.input_dir, mode=mode, device=device)


    # Obtain the initial mesh and compute its connectivity
    mesh_initial: Mesh = None

    # Use args.initial_mesh as path to the mesh
    print(f"Adjusting mesh {args.initial_mesh} direction and scale! (out of place)")
    mesh_ = adjust_and_scale_obj(args.initial_mesh, args.bound)

    vertices = np.array(mesh_.vertices, dtype=np.float32)
    indices = np.array(mesh_.faces, dtype=np.int32)
    mesh_initial = Mesh(vertices, indices, device)

    mesh_initial.compute_connectivity()


    aabb = AABB(mesh_initial.vertices.cpu().numpy())
    aabb.save(experiment_dir / "bbox.txt")

    # Apply the normalizing affine transform, which maps the bounding box to 
    # a 2-cube centered at (0, 0, 0), to the views, the mesh, and the bounding box
    space_normalization = SpaceNormalization(aabb.corners)
    views = space_normalization.normalize_views(views)
    mesh_initial = space_normalization.normalize_mesh(mesh_initial)
    aabb = space_normalization.normalize_aabb(aabb)

    # Configure the renderer
    renderer = Renderer(device=device)
    renderer.set_near_far(views, torch.from_numpy(aabb.corners).to(device), epsilon=0.5)


    # Configure the view sampler
    # view_sampler_all = ViewSampler(views=views, mode='all', views_per_iter=1)
    view_sampler_all = ViewSampler(views=views, mode='several', views_per_iter=1, picked_views=list(range(args.picked_views_first[0], args.picked_views_first[1])))
    view_sampler_several = ViewSampler(views=views, mode='several', views_per_iter=1, picked_views=args.picked_views_second)


    # Create the optimizer for the vertex positions 
    # (we optimize offsets from the initial vertex position)
    lr_vertices = args.lr_vertices
    vertex_offsets = nn.Parameter(torch.zeros_like(mesh_initial.vertices))
    args.optim_only_visible = False
    if not args.optim_only_visible:
        optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)

    # Create the optimizer for the neural shader
    shader = NeuralShader(hidden_features_layers=args.hidden_features_layers,
                          hidden_features_size=args.hidden_features_size,
                          fourier_features=args.fourier_features,
                          activation=args.activation,
                          fft_scale=args.fft_scale,
                          last_activation=torch.nn.Sigmoid, 
                          device=device)
    optimizer_shader = torch.optim.Adam(shader.parameters(), lr=args.lr_shader)

    # Initialize the loss weights and losses
    loss_weights_first = {
        "mask": 2,
        "normal_consistency": 0.1,
        "laplacian": 800,
    }
    losses_first = {k: torch.tensor(0.0, device=device) for k in loss_weights_first}

    loss_weights_second = {
        "hole_mask": args.weight_hole_mask,
        "mask": args.weight_mask,
        "normal_consistency": args.weight_normal_consistency,
        "laplacian": args.weight_laplacian,
        "normal": args.weight_normal,
        "shading": args.weight_shading
    }
    losses_second = {k: torch.tensor(0.0, device=device) for k in loss_weights_second}

    progress_bar_first = tqdm(range(1, args.iterations_first + 1))
    progress_bar_second = tqdm(range(args.iterations_first + 1, args.iterations_second + args.iterations_first + 1))



    print(' ')
    print("=============== Start First Stage =================")
    print(' ')

    for iteration in progress_bar_first:
        progress_bar_first.set_description(desc=f'Iteration {iteration}')

        # Deform the initial mesh
        mesh = mesh_initial.with_vertices(mesh_initial.vertices + vertex_offsets)

        # Sample a view subset
        views_subset = view_sampler_all()

         # find vertices visible from image views
        if args.optim_only_visible:
            vis_mask = renderer.get_vert_visibility(views_subset, mesh)
            target_vertices = nn.Parameter(vertex_offsets[vis_mask].clone())
            detach_vertices = vertex_offsets[~vis_mask].detach()
            optimizer_vertices = torch.optim.Adam([target_vertices], lr=lr_vertices)
            
            mesh_vertices = mesh_initial.vertices.detach().clone()
            mesh_vertices[vis_mask] += target_vertices  
            mesh_vertices[~vis_mask] += detach_vertices
            mesh = mesh_initial.with_vertices(mesh_vertices)

        # Render the mesh from the views
        # Perform antialiasing here because we cannot antialias after shading if we only shade a some of the pixels
        gbuffers = renderer.render(views_subset, mesh, channels=['mask', 'position', 'normal'], with_antialiasing=True) 

        # Combine losses and weights
        if loss_weights_first['mask'] > 0:
            losses_first['mask'] = mask_loss(views_subset, gbuffers)
        if loss_weights_first['normal_consistency'] > 0:
            losses_first['normal_consistency'] = normal_consistency_loss(mesh)
        if loss_weights_first['laplacian'] > 0:
            losses_first['laplacian'] = laplacian_loss(mesh)
                
        loss = torch.tensor(0., device=device)
        for k, v in losses_first.items():
            loss += v * loss_weights_first[k]

        # Optimize
        optimizer_vertices.zero_grad()
        loss.backward()
        optimizer_vertices.step()

        if args.optim_only_visible:
            vertex_offsets = torch.zeros_like(vertex_offsets)
            vertex_offsets[vis_mask] = target_vertices
            vertex_offsets[~vis_mask] = detach_vertices

        progress_bar_first.set_postfix({'loss': loss.detach().cpu()})

        # Visualizations
        if (args.visualization_frequency > 0) and (iteration == 1 or iteration % args.visualization_frequency == 0):
            import matplotlib.pyplot as plt
            with torch.no_grad():
                use_fixed_views = len(args.visualization_views) > 0
                view_indices = args.visualization_views if use_fixed_views else [np.random.choice(list(range(len(views_subset))))]
                for vi in view_indices:
                    try:
                        debug_view = views[vi] if use_fixed_views else views_subset[vi]
                        debug_gbuffer = renderer.render([debug_view], mesh, channels=['mask', 'position', 'normal'], with_antialiasing=True)[0]

                        position = debug_gbuffer["position"]
                        normal = debug_gbuffer["normal"]
                        mask = debug_gbuffer["mask"]
                        view_direction = torch.nn.functional.normalize(debug_view.camera.center - position, dim=-1)

                        # Save a normal map in camera space
                        normal_path = (images_save_path / str(vi) / "normal") if use_fixed_views else (images_save_path / "normal")
                        normal_path.mkdir(parents=True, exist_ok=True)
                        R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)
                        normal_image = (0.5*(normal @ debug_view.camera.R.T @ R.T + 1)) * mask + (1-mask) # camera-view normal, range(0-1) RGB
                        # normal_image = (0.5*(normal + 1)) * debug_gbuffer["mask"] + (1-debug_gbuffer["mask"])  # global normal
                        plt.imsave(normal_path / f'normal_{iteration}.png', normal_image.cpu().numpy()) # must save plt in range (0-1)!


                        # Save Estimated normal image
                        estimated_normal_path = (images_save_path / str(vi) / "Estimated_normal") if use_fixed_views else (images_save_path / "Estimated_normal")
                        estimated_normal_path.mkdir(parents=True, exist_ok=True)
                        estimated_normal = debug_view.normal.detach()
                        estimated_normal_image = estimated_normal * debug_view.mask + (1 - debug_view.mask)
                        plt.imsave(os.path.join(estimated_normal_path / f'estimated_normal_{iteration}.png'), estimated_normal_image.cpu().numpy()) # range(0-1) RGB
                    
                    except ValueError as e:
                        print(f"Error saving image at iteration {iteration}: {e}, but we are gonna ignore it")

        if (args.save_frequency > 0) and (iteration == 1 or iteration % args.save_frequency == 0):
            with torch.no_grad():
                mesh_for_writing = space_normalization.denormalize_mesh(mesh.detach().to('cpu'))
                write_mesh(meshes_save_path / f"mesh_{iteration:06d}.obj", mesh_for_writing)                                



# =============================================================================================
# =============================================================================================
# =============================================================================================

    # reference mesh
    rf_mesh = mesh_initial.with_vertices(mesh_initial.vertices.clone().detach().requires_grad_(False) + vertex_offsets.clone().detach().requires_grad_(False))

    args.optim_only_visible = True
    print(' ')
    print("=============== Start Second Stage =================")
    for iteration in progress_bar_second:
        progress_bar_second.set_description(desc=f'Iteration {iteration}')

        if iteration in args.upsample_iterations:
            # Upsample the mesh by remeshing the surface with half the average edge length
            print(f"=== Remesh at iteration == {iteration}")
            e0, e1 = mesh.edges.unbind(1)
            average_edge_length = torch.linalg.norm(mesh.vertices[e0] - mesh.vertices[e1], dim=-1).mean()
            
            v_upsampled, f_upsampled = gpytoolbox.remesh_botsch(mesh.vertices.detach().cpu().numpy().astype(np.float64), mesh.indices.detach().cpu().numpy().astype(np.int32), h=float(average_edge_length/2))
            v_upsampled = np.ascontiguousarray(v_upsampled)
            f_upsampled = np.ascontiguousarray(f_upsampled)

            mesh_initial = Mesh(v_upsampled, f_upsampled, device=device)
            mesh_initial.compute_connectivity()

            # Adjust weights and step size
            loss_weights_second['laplacian'] *= 4
            loss_weights_second['normal_consistency'] *= 4
            lr_vertices *= 0.25
            loss_weights_second['normal'] *= 1

            # Create a new optimizer for the vertex offsets
            vertex_offsets = nn.Parameter(torch.zeros_like(mesh_initial.vertices))
            if not args.optim_only_visible:
                optimizer_vertices = torch.optim.Adam([vertex_offsets], lr=lr_vertices)     

        # Deform the initial mesh
        mesh = mesh_initial.with_vertices(mesh_initial.vertices + vertex_offsets)

        # Sample a view subset
        views_subset = view_sampler_several()

        # find vertices visible from image views
        if args.optim_only_visible:
            vis_mask = renderer.get_vert_visibility(views_subset, mesh)
            target_vertices = nn.Parameter(vertex_offsets[vis_mask].clone())
            detach_vertices = vertex_offsets[~vis_mask].detach()
            optimizer_vertices = torch.optim.Adam([target_vertices], lr=lr_vertices)
            
            mesh_vertices = mesh_initial.vertices.detach().clone()
            mesh_vertices[vis_mask] += target_vertices  
            mesh_vertices[~vis_mask] += detach_vertices
            mesh = mesh_initial.with_vertices(mesh_vertices)

        # Render the mesh from the views
        # Perform antialiasing here because we cannot antialias after shading if we only shade a some of the pixels
        gbuffers = renderer.render(views_subset, mesh, channels=['mask', 'position', 'normal'], with_antialiasing=True) 
        with torch.no_grad():
            gbuffers_rf = renderer.render(views_subset, rf_mesh, channels=['mask', 'position', 'normal'], with_antialiasing=True) 


        # Combine losses and weights
        if loss_weights_second['hole_mask'] > 0:
            losses_second['hole_mask'] = hole_mask_loss(views_subset, gbuffers, gbuffers_rf)
        if loss_weights_second['mask'] > 0:
            losses_second['mask'] = mask_loss(views_subset, gbuffers)
        if loss_weights_second['normal_consistency'] > 0:
            losses_second['normal_consistency'] = normal_consistency_loss(mesh)
        if loss_weights_second['laplacian'] > 0:
            losses_second['laplacian'] = laplacian_loss(mesh)
        if loss_weights_second['normal'] > 0:
            if args.enhanced_normal_map_loss:
                losses_second['normal'] = normal_map_loss_enhanced(views_subset, gbuffers)
            else:
                losses_second['normal'] = normal_map_loss(views_subset, gbuffers)
        if loss_weights_second['shading'] > 0:
            losses_second['shading'] = shading_loss(views_subset, gbuffers, shader=shader, shading_percentage=args.shading_percentage)
        

        loss = torch.tensor(0., device=device)
        for k, v in losses_second.items():
            loss += v * loss_weights_second[k]


        # Optimize
        optimizer_vertices.zero_grad()
        optimizer_shader.zero_grad()
        loss.backward()
        optimizer_vertices.step()
        optimizer_shader.step()

        if args.optim_only_visible:
            vertex_offsets = torch.zeros_like(vertex_offsets)
            vertex_offsets[vis_mask] = target_vertices
            vertex_offsets[~vis_mask] = detach_vertices

        progress_bar_second.set_postfix({'loss': loss.detach().cpu()})

        # Visualizations
        if (args.visualization_frequency > 0) and (iteration == 1 or iteration % args.visualization_frequency == 0):
            import matplotlib.pyplot as plt
            with torch.no_grad():
                use_fixed_views = len(args.visualization_views) > 0
                view_indices = args.visualization_views if use_fixed_views else [np.random.choice(list(range(len(views_subset))))]
                for vi in view_indices:
                    try:

                        debug_view = views[vi] if use_fixed_views else views_subset[vi]
                        debug_gbuffer = renderer.render([debug_view], mesh, channels=['mask', 'position', 'normal'], with_antialiasing=True)[0]

                        position = debug_gbuffer["position"]
                        normal = debug_gbuffer["normal"]
                        mask = debug_gbuffer["mask"]
                        view_direction = torch.nn.functional.normalize(debug_view.camera.center - position, dim=-1)

                        # Save a normal map in camera space
                        normal_path = (images_save_path / str(vi) / "normal") if use_fixed_views else (images_save_path / "normal")
                        normal_path.mkdir(parents=True, exist_ok=True)
                        R = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], device=device, dtype=torch.float32)
                        normal_image = (0.5*(normal @ debug_view.camera.R.T @ R.T + 1)) * mask + (1-mask) # camera-view normal, range(0-1) RGB
                        # normal_image = (0.5*(normal + 1)) * debug_gbuffer["mask"] + (1-debug_gbuffer["mask"])  # global normal
                        plt.imsave(normal_path / f'normal_{iteration}.png', normal_image.detach().cpu().numpy()) # must save plt in range (0-1)!
                        


                        # Save Estimated normal image
                        estimated_normal_path = (images_save_path / str(vi) / "Estimated_normal") if use_fixed_views else (images_save_path / "Estimated_normal")
                        estimated_normal_path.mkdir(parents=True, exist_ok=True)
                        estimated_normal = debug_view.normal.detach()
                        estimated_normal_image = estimated_normal * debug_view.mask + (1 - debug_view.mask)
                        plt.imsave(os.path.join(estimated_normal_path / f'estimated_normal_{iteration}.png'), estimated_normal_image.cpu().numpy()) # range(0-1) RGB

                        # Save the shaded rendering
                        shaded_image = shader(position, normal, view_direction) * debug_gbuffer["mask"] + (1-debug_gbuffer["mask"])
                        shaded_path = (images_save_path / str(vi) / "shaded") if use_fixed_views else (images_save_path / "shaded")
                        shaded_path.mkdir(parents=True, exist_ok=True)
                        plt.imsave(shaded_path / f'neuralshading_{iteration}.png', shaded_image.cpu().numpy())
                    
                    except ValueError as e:
                        print(f"Error saving image at iteration {iteration}: {e}, but we are gonna ignore it")


        if (args.save_frequency > 0) and (iteration == 1 or iteration % args.save_frequency == 0):
            with torch.no_grad():
                mesh_for_writing = space_normalization.denormalize_mesh(mesh.detach().to('cpu'))
                write_mesh(meshes_save_path / f"mesh_{iteration:06d}.obj", mesh_for_writing)  
            # shader.save(shaders_save_path / f'shader_{iteration:06d}.pt')
                              

    final_mesh = space_normalization.denormalize_mesh(mesh.detach().to('cpu'))
    write_mesh(meshes_save_path / f"mesh_{args.iterations_first + args.iterations_second:06d}.obj", final_mesh, post_process=False)

    # post process
    write_mesh(os.path.join(os.path.dirname(args.output_dir), 'final_mesh.obj'), final_mesh, post_process=True)
    
    # if shader is not None:
    #     shader.save(shaders_save_path / f'shader_{args.iterations_first + args.iterations_second:06d}.pt')

    return os.path.join(os.path.dirname(args.output_dir), 'final_mesh.obj')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir', type=Path, default="./data", help="Path to the input data")
    # parser.add_argument('--output_dir', type=Path, default="./out", help="Path to the output directory")
    # parser.add_argument('--initial_mesh', type=str, default="vh32", help="Initial mesh, either a path or one of [vh16, vh32, vh64, sphere16]")
    # parser.add_argument('--run_name', type=str, default=None, help="Name of this run")


    # parser.add_argument('--enhanced_normal_map_loss', type=bool, default=True)
    # parser.add_argument('--optim_only_visible', type=bool, default=True)
    # parser.add_argument('--iterations_first', type=int, default=3000, help="Total number of iterations in first stage")
    # parser.add_argument('--iterations_second', type=int, default=1000, help="Total number of iterations in second stage")
    # parser.add_argument('--upsample_iterations', type=list, default=[3500])


    # parser.add_argument('--lr_vertices', type=float, default=1e-3, help="Step size/learning rate for the vertex positions")

    # parser.add_argument('--save_frequency', type=int, default=100, help="Frequency of mesh and shader saving (in iterations)")
    # parser.add_argument('--visualization_frequency', type=int, default=100, help="Frequency of shader visualization (in iterations)")
    # parser.add_argument('--visualization_views', type=int, nargs='+', default=[], help="Views to use for visualization. By default, a random view is selected each time")
    # parser.add_argument('--device', type=int, default=0, choices=([-1] + list(range(torch.cuda.device_count()))), help="GPU to use; -1 is CPU")

    # # loss weights for second stage
    # parser.add_argument('--weight_hole_mask', type=float, default=2.0, help="Weight of the hole mask loss term")
    # parser.add_argument('--weight_mask', type=float, default=2.0, help="Weight of the mask term")
    # parser.add_argument('--weight_normal_consistency', type=float, default=0.1, help="Weight of the normal term")
    # parser.add_argument('--weight_laplacian', type=float, default=40.0, help="Weight of the laplacian term")
    # parser.add_argument('--weight_normal', type=float, default=0.8, help="Weight of the normal term")


    # # shading related args
    # parser.add_argument('--lr_shader', type=float, default=1e-3, help="Step size/learning rate for the shader parameters")
    # parser.add_argument('--weight_shading', type=float, default=1.0, help="Weight of the shading term")
    # parser.add_argument('--shading_percentage', type=float, default=0.75, help="Percentage of valid pixels considered in the shading loss (0-1)")
    # parser.add_argument('--hidden_features_layers', type=int, default=3, help="Number of hidden layers in the positional feature part of the neural shader")
    # parser.add_argument('--hidden_features_size', type=int, default=256, help="Width of the hidden layers in the neural shader")
    # parser.add_argument('--fourier_features', type=str, default='positional', choices=(['none', 'gfft', 'positional']), help="Input encoding used in the neural shader")
    # parser.add_argument('--activation', type=str, default='relu', choices=(['relu', 'sine']), help="Activation function used in the neural shader")
    # parser.add_argument('--fft_scale', type=int, default=4, help="Scale parameter of frequency-based input encodings in the neural shader")