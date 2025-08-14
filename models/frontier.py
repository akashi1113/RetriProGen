import torch
from torch.nn import Module, Sequential
from torch.nn import functional as F

from .utils import GVPerceptronVN, GVLinear


class FrontierLayerVN(Module):
    def __init__(self, in_sca, in_vec, hidden_dim_sca, hidden_dim_vec):
        super().__init__()
        self.net = Sequential(
            GVPerceptronVN(in_sca, in_vec, hidden_dim_sca, hidden_dim_vec),
            GVLinear(hidden_dim_sca, hidden_dim_vec, 256, 256)
        )

    def forward(self, h_att, idx=None):
        if idx is None:
            h_att_a = [h_att[0], h_att[1]]
        else:
            h_att_a = [h_att[0][idx], h_att[1][idx]]
        pred_sca, pred_vec = self.net(h_att_a)
        return pred_sca
        # results = []
        # for i in range(h_att[0].size(0)):
        #     e1 = h_att[0][i]
        #     e2 = h_att[1][i]
        #     h_att_ligand = [e1, e2]
        #     pred_sca, pred_vec = self.net(h_att_ligand)
        #     pred_sca = pred_sca.to(h_att[0].device)
        #     pred_vec = pred_vec.to(h_att[1].device)
        #     pred_sca_expanded = pred_sca.unsqueeze(-1)  # (83, 256, 1)
        #     pred = pred_sca_expanded + pred_vec  # (83, 256, 3)
        #
        #     results.append(pred)
        # output_tensor = torch.stack(results, dim=0)
        # return output_tensor
