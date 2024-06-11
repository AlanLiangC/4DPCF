from .al1_model import AL1_Model
from .ori_model import Ori_Model
from .al3_model import AL3_Model
from .lstm_model import LSTM_Model
from .scene_model import Scene_Model
from .rt_model import RayTracing_Model

model_maps = {
    'al1': AL1_Model,
    'ori': Ori_Model,
    'lstm': LSTM_Model,
    'rt': RayTracing_Model,
    'scene': Scene_Model,
    'al3': AL3_Model
}

__all__ = ['model_maps', 'AL1_Model', 'Ori_Model', 'AL3_Model', 'LSTM_Model', 'Scene_Model', 'RayTracing_Model']