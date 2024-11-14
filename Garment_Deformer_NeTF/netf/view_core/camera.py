import torch
import numpy as np

def perspective(fovy, near=0.01, far=100):
    """get the perspective matrix

    Returns:
        np.ndarray: camera perspective, float [4, 4]
    """

    y = np.tan(fovy / 2)
    aspect = 1
    return np.array(
        [
            [1 / (y * aspect), 0, 0, 0],
            [0, -1 / y, 0, 0],
            [
                0,
                0,
                -(far + near) / (far - near),
                -(2 * far * near) / (far - near),
            ],
            [0, 0, -1, 0],
        ],
        dtype=np.float32,
    )

class Camera:
    def __init__(self, K, C2W):
        self.K = K
        self.C2W = C2W
    
