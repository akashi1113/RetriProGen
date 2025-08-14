import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, LeakyReLU
import numpy as np
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from math import pi as PI
EPS = 1e-6
# from utils.profile import lineprofile


class MessageModule(Module):
    def __init__(self, node_sca, node_vec, edge_sca, edge_vec, out_sca, out_vec, cutoff=10.):
        super().__init__()
        hid_sca, hid_vec = edge_sca, edge_vec
        self.cutoff = cutoff
        self.node_gvlinear = GVLinear(node_sca, node_vec, out_sca, out_vec)
        self.edge_gvp = GVPerceptronVN(edge_sca, edge_vec, hid_sca, hid_vec)

        self.sca_linear = Linear(hid_sca, out_sca)  # edge_sca for y_sca
        self.e2n_linear = Linear(hid_sca, out_vec)
        self.n2e_linear = Linear(out_sca, out_vec)
        self.edge_vnlinear = VNLinear(hid_vec, out_vec)

        self.out_gvlienar = GVLinear(out_sca, out_vec, out_sca, out_vec)

    def forward(self, node_features, edge_features, edge_index_node,dist_ij=None, annealing=False):
        node_scalar, node_vector = self.node_gvlinear(node_features)#只保留了蛋白的
        node_scalar, node_vector = node_scalar[edge_index_node], node_vector[edge_index_node]

        edge_scalar, edge_vector = self.edge_gvp(edge_features)#57030条边，只保留蛋白信息

        y_scalar = node_scalar * self.sca_linear(edge_scalar)
        y_node_vector = self.e2n_linear(edge_scalar).unsqueeze(-1) * node_vector
        y_edge_vector = self.n2e_linear(node_scalar).unsqueeze(-1) * self.edge_vnlinear(edge_vector)
        y_vector = y_node_vector + y_edge_vector

        output = self.out_gvlienar((y_scalar, y_vector))

        if annealing:
            C = 0.5 * (torch.cos(dist_ij * PI / self.cutoff) + 1.0)  # (A, 1)
            C = C * (dist_ij <= self.cutoff) * (dist_ij >= 0.0)
            output = [output[0] * C.view(-1, 1), output[1] * C.view(-1, 1, 1)]   # (A, 1)
        return output


class GVPerceptronVN(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        self.gv_linear = GVLinear(in_scalar, in_vector, out_scalar, out_vector)
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(out_vector)

    def forward(self, x):
        sca_in, vec_in = x
        sca, vec = self.gv_linear(x)
        sca=sca.to(sca_in.device)
        vec=vec.to(vec_in.device)
        vec = self.act_vec(vec)
        sca = self.act_sca(sca)
        return sca, vec


class GVLinear(Module):
    def __init__(self, in_scalar, in_vector, out_scalar, out_vector):
        super().__init__()
        dim_hid = max(in_vector, out_vector)
        self.lin_vector = VNLinear(in_vector, dim_hid, bias=False)
        self.lin_vector2 = VNLinear(dim_hid, out_vector, bias=False)
        # self.group_lin_vector = VNGroupLinear(dim_hid, out_vector, bias=False)
        # self.group_lin_scalar = Conv1d(in_scalar + dim_hid, out_scalar, 1, bias=False)
        self.scalar_to_vector_gates = Linear(out_scalar, out_vector)
        self.lin_scalar = Linear(in_scalar + dim_hid, out_scalar, bias=False)

    def forward(self, features):
        feat_scalar, feat_vector = features
        feat_scalar = feat_scalar.to(feat_vector.device)  # 将标量部分移动到和向量部分相同的设备上
        feat_vector = feat_vector.to(feat_vector.device)  # 将向量部分移动到和标量部分相同的设备上
        feat_vector_inter = self.lin_vector(feat_vector)  # (N_samples, dim_hid, 3)
        feat_vector_norm = torch.norm(feat_vector_inter, p=2, dim=-1)  # (N_samples, dim_hid)
        feat_vector_norm = feat_vector_norm.to(feat_scalar.device)
        feat_scalar_cat = torch.cat([feat_vector_norm, feat_scalar], dim=-1)  # (N_samples, dim_hid+in_scalar)
        feat_scalar_cat = feat_scalar_cat.to(self.lin_scalar.weight.device)

        out_scalar = self.lin_scalar(feat_scalar_cat)
        out_vector = self.lin_vector2(feat_vector_inter)

        gating = torch.sigmoid(self.scalar_to_vector_gates(out_scalar)).unsqueeze(dim = -1)
        out_vector = gating * out_vector
        return out_scalar, out_vector


class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(VNLinear, self).__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, *args, **kwargs)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, N_feat, 3]
        '''
        x = x.to(self.map_to_feat.weight.device)
        x_out = self.map_to_feat(x.transpose(-2,-1)).transpose(-2,-1)
        x_out = x_out.to(x.device)
        return x_out


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.01):
        super(VNLeakyReLU, self).__init__()
        if share_nonlinearity == True:
            self.map_to_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, N_feat, 3]
        '''
        x = x.to(self.map_to_dir.weight.device)
        d = self.map_to_dir(x.transpose(-2,-1)).transpose(-2,-1)  # (N_samples, N_feat, 3)
        dotprod = (x*d).sum(-1, keepdim=True)  # sum over 3-value dimension
        mask = (dotprod >= 0).to(x.dtype)
        d_norm_sq = (d*d).sum(-1, keepdim=True)  # sum over 3-value dimension
        x_out = (self.negative_slope * x +
                (1-self.negative_slope) * (mask*x + (1-mask)*(x-(dotprod/(d_norm_sq+EPS))*d)))

        return x_out


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)
