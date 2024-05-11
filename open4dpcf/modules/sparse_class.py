import torch.nn as nn
from functools import partial

import spconv
if float(spconv.__version__[2:]) >= 2.2:
    spconv.constants.SPCONV_USE_DIRECT_TABLE = False
    
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

class SparseClass(nn.Module):

    @staticmethod
    def to_sparse_tensor(voxel_feature,
                         coords,
                         sparse_shape,
                         batch_size):
        
        x = spconv.SparseConvTensor(voxel_feature, 
                                    coords,
                                    sparse_shape, 
                                    batch_size)
        
        return x
