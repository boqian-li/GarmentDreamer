import trimesh
import numpy as np

def adjust_and_scale_obj(input_file, bound):
    # 加载OBJ文件
    mesh = trimesh.load_mesh(input_file, process=False, maintain_order=True)
    
    # 获取顶点坐标
    vertices = mesh.vertices
    
    # 调整顶点坐标
    adjustment = np.zeros_like(vertices)
    adjustment[:,0] = vertices[:,2]
    adjustment[:,1] = vertices[:,0]
    adjustment[:,2] = vertices[:,1]
    
    # 缩放顶点坐标
    scaled_adjustment = adjustment * bound
    
    # 更新mesh的顶点坐标
    mesh.vertices = scaled_adjustment
    
    return mesh

# 用法示例
if __name__ == "__main__":
    input_file = "../load/mesh/Dress_1885_simulated.obj"  # 输入OBJ文件路径
    output_file = "../load/mesh/Dress_1885_simulated_adjusted.obj"  # 输出OBJ文件路径
    adjust_and_scale_obj(input_file, output_file)
