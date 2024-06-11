import torch
import numpy as np
from plyfile import PlyData, PlyElement
import matplotlib.cm as cm
import matplotlib as mpl
from io import BytesIO
from .sh_utils import RGB2SH

def normalize_vector(vector):
    v_max = np.max(vector)
    v_min = np.min(vector)
    return (vector - v_min) / (v_max - v_min + 1e-5)

def vector2rgb(vector: np.ndarray):

    norm = mpl.colors.Normalize(vmin=-0.1, vmax=1.1)
    cmap = cm.jet  # sequential
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = m.to_rgba(normalize_vector(vector))
    colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]

    return colors[:,:3]

def tensor2ndarray(data):

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return data

def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    for i in range(45):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l

def gaussian_sh_color(rgb, max_sh_degree = 3):
    fused_color = RGB2SH(rgb)
    features = np.zeros((fused_color.shape[0], 3, (max_sh_degree + 1) ** 2))
    features[:, :3, 0 ] = fused_color
    features[:, 3:, 1:] = 0.0
    sh_dc = features[:,:,0:1].transpose(0, 2, 1).reshape(-1,3)
    sh_rest = features[:,:,1:].transpose(0, 2, 1).reshape(-1,45)
    return sh_dc, sh_rest

def point2gaussians_ply(points, colors, path: str = None, cus_scale = 0.5):
    '''
    points : Nx3,
    colors : N
    '''

    xyz = tensor2ndarray(points) # 3
    _colors = tensor2ndarray(colors) # 3
    rgb = vector2rgb(_colors)
    sh_dc, sh_rest = gaussian_sh_color(rgb)
    normals = np.zeros_like(xyz) # 3

    opacities = np.ones((rgb.shape[0],1)) * 15.26 # 1
    scale = np.ones_like(xyz) * np.log(cus_scale)
    rotation = np.zeros([xyz.shape[0], 4])
    rotation[:,0] = 1

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, sh_dc, sh_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def point2gaussians_splat(points, colors, path: str = None, cus_scale = 1):
    '''
    points : Nx3,
    colors : N
    '''
    xyz = tensor2ndarray(points) # 3
    _colors = tensor2ndarray(colors) # 3
    rgb = vector2rgb(_colors)

    buffer = BytesIO()
    for i in range(points.shape[0]):
        position = xyz[i]
        scales = np.exp(
            np.array(
                [cus_scale] * 3,
                dtype=np.float32,
            )
        ) 
        rot = np.array(
            [1,0,0,0],
            dtype=np.float32,
        )
        color = np.array(
            [
                rgb[i][0],
                rgb[i][1],
                rgb[i][2],
                1 / (1 + np.exp(-1)),
            ]
        )

        buffer.write(position.tobytes())
        buffer.write(scales.tobytes())
        buffer.write((color * 255).clip(0, 255).astype(np.uint8).tobytes())
        buffer.write(
            ((rot / np.linalg.norm(rot)) * 128 + 128)
            .clip(0, 255)
            .astype(np.uint8)
            .tobytes()
        )

    splat_data = buffer.getvalue()
    with open(path, "wb") as f:
        f.write(splat_data)
