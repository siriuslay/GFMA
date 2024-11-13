from typing import Optional, List, Tuple, Union

"""Contains novel layer definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter
import dgl.function as fn
from dgl.base import DGLError
from torch.nn import init
from dgl.utils import expand_as_pair
from torch.nn.modules import Module
import math
DEFAULT_THRESHOLD = None


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self):
        super(Binarizer, self).__init__()

    @staticmethod
    def forward(self, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.lt(threshold)] = 0
        outputs[inputs.ge(threshold)] = 1
        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return (gradOutput, None)


class MaskGraphConv(nn.Module):
    """Modified GraphConv with masks for weights."""

    def __init__(
        self,
        in_feats,
        out_feats,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
        mask_init='1s', mask_scale=1e-2,
        threshold_fn='binarizer', threshold=None, apply_mask=True,
        dev=0
    ):
        super(MaskGraphConv, self).__init__()
        # if norm not in ("none", "both", "right", "left"):
        #     raise DGLError(
        #         'Invalid norm value. Must be either "none", "both", "right" or "left".'
        #         ' But got "{}".'.format(norm)
        #     )
        self.dev = dev
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.apply_mask = apply_mask
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }
        # weight and bias are no longer Parameters.
        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats), requires_grad=False)
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats), requires_grad=False)  # .to(f"cuda:{self.dev}")
        else:
            self.register_parameter("bias", None)

        self._activation = activation

        if apply_mask:
            # Initialize real-valued mask weights.
            self.mask_real = self.weight.data.new(self.weight.size())
            self.threshold = threshold
            if mask_init == '1s':
                self.mask_real.fill_(mask_scale)  # 每个元素填充mask_scale
            elif mask_init == 'uniform':
                self.mask_real.uniform_(0, mask_scale)  # 每个元素填充0~mask_scale直接的任意值
                # self.mask_real.uniform_(-1 * mask_scale, mask_scale)
            # mask_real is now a trainable parameter.
            self.mask_real = Parameter(self.mask_real)
        else:
            self.mask_real = self.weight.data.new(self.weight.size())
            self.threshold = threshold
            if mask_init == '1s':
                self.mask_real.fill_(mask_scale)  # 每个元素填充mask_scale
            elif mask_init == 'uniform':
                self.mask_real.uniform_(0, mask_scale)  # 每个元素填充0~mask_scale直接的任意值
                # self.mask_real.uniform_(-1 * mask_scale, mask_scale)
            # mask_real is now a trainable parameter.
            self.mask_real = Parameter(self.mask_real, requires_grad=False)

        self.soft = False
        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer().apply

    def set_allow_zero_in_degree(self, set_value):

        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None, edge_weight=None):

        with graph.local_scope():
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            # if torch.isnan(feat).any():
            #     print('feat_src is nan')
            feat_src, feat_dst = expand_as_pair(feat, graph)
            # if torch.isnan(feat_src).any():
            #     print('feat_src is nan')
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                if self.apply_mask:
                    # Get binarized/ternarized mask from real-valued mask.
                    if self.soft:
                        mask_thresholded = self.threshold_fn(self.mask_real, "simple")
                    elif self.training:
                        # mask_thresholded = self.threshold_fn(self.mask_real, self.threshold)
                        mask_thresholded = ((self.mask_real > self.threshold).float() - self.mask_real).detach() + self.mask_real
                        print(torch.sum(mask_thresholded))
                    else:
                        mask_thresholded = (self.mask_real > self.threshold)  # .float()
                    # Mask weights with above mask.
                    weight_thresholded = mask_thresholded * self.weight
                    # weight_thresholded = weight_thresholded.to(f"cuda:{self.dev}")

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight_thresholded is not None:
                    feat_src = torch.matmul(feat_src, weight_thresholded)
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
                # if torch.isnan(rst).any():
                #     print('before degs, rst is nan')
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
                if weight_thresholded is not None:
                    rst = torch.matmul(rst, weight_thresholded)

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                # if torch.any(degs == 0):
                #     print('degs = 0')
                if self._norm == "both":
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}, out={_out_feats}"
        summary += ", normalization={_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)


class MaskLinear(Module):
    """Modified Linear with masks for weights."""

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 dev=0, dtype=None, mask_init='1s', mask_scale=1e-2,
        threshold_fn='binarizer', threshold=None, apply_mask=True) -> None:
        factory_kwargs = {'device': torch.device(f"cuda:{dev}" if torch.cuda.is_available() else "cpu"), 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.apply_mask = apply_mask
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }
        if apply_mask:
            # Initialize real-valued mask weights.
            self.mask_real = self.weight.data.new(self.weight.size())
            self.threshold = threshold
            if mask_init == '1s':
                self.mask_real.fill_(mask_scale)  # 每个元素填充mask_scale
            elif mask_init == 'uniform':
                self.mask_real.uniform_(0, mask_scale)  # 每个元素填充0~mask_scale直接的任意值
                # self.mask_real.uniform_(-1 * mask_scale, mask_scale)
            # mask_real is now a trainable parameter.
            self.mask_real = Parameter(self.mask_real)

            self.soft = False
            # Initialize the thresholder.
            if threshold_fn == 'binarizer':
                self.threshold_fn = Binarizer().apply

    def forward(self, input: Tensor) -> Tensor:
        if self.apply_mask:
            # self.mask_real.data = (self.mask_real.data ** 2) / (self.mask_real.data ** 2 + 1)
            # Get binarized/ternarized mask from real-valued mask.
            if self.soft:
                mask_thresholded = self.mask_real
            elif self.training:
                mask_thresholded = self.mask_real
                # mask_thresholded = self.threshold_fn(self.mask_real, self.threshold)
                print(torch.sum(mask_thresholded))
                # mask_thresholded = ((self.mask_real >= self.threshold).float() - self.mask_real).detach() + self.mask_real
            else:
                mask_thresholded = self.mask_real
                # mask_thresholded = (self.mask_real >= self.threshold)  # .float()
            # Mask weights with above mask.
            weight_thresholded = mask_thresholded * self.weight
        return F.linear(input, weight_thresholded, self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

