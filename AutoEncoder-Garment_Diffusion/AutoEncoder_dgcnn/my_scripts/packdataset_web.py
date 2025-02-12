import os
import webdataset as wds
import numpy as np
import torch
from tqdm import tqdm

def pack_images_to_webdataset(input_folder, output_folder, tar_file_prefix, max_images_per_tar=1000):
    assert os.path.isdir(input_folder), "Input folder not found."
    os.makedirs(output_folder, exist_ok=False)

    # Define the pattern for naming the output TAR files
    tar_pattern = os.path.join(output_folder, f"{tar_file_prefix}-%06d.tar")

    # Create a writer for WebDataset
    sink = wds.ShardWriter(tar_pattern, maxcount=max_images_per_tar)

    for filename in tqdm(os.listdir(input_folder)):
        if filename.endswith('.npz'):
            file_path = os.path.join(input_folder, filename)
            npz = np.load(file_path)
            vertices = npz['vertices']
            triangles = npz['triangles']
            pcds=npz['pcds']
            selected_coords=npz['selected_coords']
            selected_gt_udf=npz['selected_gt_udf']
            selected_gt_grad=npz['selected_gt_grad']

            # Write the data to the current tar shard
            sink.write({
                "__key__": f"{filename[:-4]}",
                "vertices.npy": vertices,
                "triangles.npy": triangles,
                "pcds.npy": pcds,
                "selected_coords.npy": selected_coords,
                "selected_gt_udf.npy": selected_gt_udf,
                "selected_gt_grad.npy": selected_gt_grad
            })

    sink.close()

# Define your paths and parameters
input_folder = '/devdata/123'
output_folder = '/devdata/zip_123'
tar_file_prefix = 'clothdataset'  # Prefix for the output TAR files
max_images_per_tar = 400  # Max number of images per TAR file

# Execute the function
pack_images_to_webdataset(input_folder, output_folder, tar_file_prefix, max_images_per_tar)