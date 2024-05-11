from .mamba_stl import MambaSTL
from .simvp_model import SimVP_Model

stl_models_map = {
    'mamba': MambaSTL,
    'simvp': SimVP_Model
}

__all__ = ['stl_models_map', 'MambaSTL', 'SimVP_Model']