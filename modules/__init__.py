from .mask_model import MaskGCN5_64, MaskGCN5_32, MaskGCN3_64, MaskGCN3_32, MaskGCN2_64, MaskGCN2_32
from .mask_model import MaskGAT5_64, MaskGAT5_32, MaskGAT3_64, MaskGAT3_32, MaskGAT2_64, MaskGAT2_32
from .mask_model import MaskGIN5_64, MaskGIN5_32, MaskGIN3_64, MaskGIN3_32, MaskGIN2_64, MaskGIN2_32


MaskGNN_dict = {
    'GCN5_64': MaskGCN5_64,
    'GCN5_32': MaskGCN5_32,
    'GCN3_64': MaskGCN3_64,
    'GCN3_32': MaskGCN3_32,
    'GCN2_64': MaskGCN2_64,
    'GCN2_32': MaskGCN2_32,

    'GAT5_64': MaskGAT5_64,
    'GAT5_32': MaskGAT5_32,
    'GAT3_64': MaskGAT3_64,
    'GAT3_32': MaskGAT3_32,
    'GAT2_64': MaskGAT2_64,
    'GAT2_32': MaskGAT2_32,

    'GIN5_64': MaskGIN5_64,
    'GIN5_32': MaskGIN5_32,
    'GIN3_64': MaskGIN3_64,
    'GIN3_32': MaskGIN3_32,
    'GIN2_64': MaskGIN2_64,
    'GIN2_32': MaskGIN2_32,
    
}