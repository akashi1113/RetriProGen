import torch
from torch.nn import Module
import torch.nn as nn
from .embedding import GVP
from .interaction import get_interaction_vn
from .frontier import FrontierLayerVN
import torch.nn.functional as F
from .duibidecoder import Decoder
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def embed_compose_GVP(compose_feature, compose_vec, idx_ligand, idx_protein,
                      ligand_atom_emb, protein_res_emb,
                      emb_dim, ligand_atom_feature=13):
    protein_nodes = (compose_feature[idx_protein], compose_vec[idx_protein])
    ligand_nodes = (
    compose_feature[idx_ligand][:, :ligand_atom_feature], compose_vec[idx_ligand][:, 0, :].unsqueeze(-2))
    h_protein = protein_res_emb(protein_nodes)
    h_ligand = ligand_atom_emb(ligand_nodes)
    #
    h_sca = torch.zeros([len(compose_feature), emb_dim[0]], ).to(h_ligand[0])
    h_vec = torch.zeros([len(compose_feature), emb_dim[1], 3], ).to(h_ligand[1])
    h_sca[idx_ligand], h_sca[idx_protein] = h_ligand[0], h_protein[0]
    h_vec[idx_ligand], h_vec[idx_protein] = h_ligand[1], h_protein[1]
    return [h_sca, h_vec]


class DuibiModel(Module):
    '''

    '''
    def __init__(self,config, protein_res_feature_dim,ligand_atom_feature_dim,num_props=None):
        super().__init__()
        self.config = config
        #得到蛋白embedding，还需要进入encoder
        self.num_props = num_props
        self.emb_dim = [config.hidden_channels, config.hidden_channels_vec]
        self.protein_res_emb = GVP(protein_res_feature_dim, self.emb_dim)
        self.ligand_atom_emb = GVP(ligand_atom_feature_dim, self.emb_dim)
        #编码器
        self.encoder= get_interaction_vn(config.encoder)
        in_sca, in_vec = self.encoder.out_sca, self.encoder.out_vec
        self.frontier_pred = FrontierLayerVN(in_sca=in_sca, in_vec=in_vec,
                                             hidden_dim_sca=128, hidden_dim_vec=32)
        self.decoder = Decoder(config.decoder,self.num_props)
        self.projection = nn.Linear(config.hidden_channels, len(config.decoder.smiVoc), bias=False)
    def forward(self,batch,smiles_index,compose_feature, compose_vec, idx_ligand, idx_protein,
                compose_pos,compose_knn_edge_index,compose_knn_edge_feature,tgt_len,prop=None):
        h_compose = embed_compose_GVP(compose_feature, compose_vec, idx_ligand, idx_protein,
                                      self.ligand_atom_emb, self.protein_res_emb, self.emb_dim)
        # h_compose[0], h_compose[1] = h_compose[0][idx_protein],h_compose[1][idx_protein]
        #a是为了提取纯蛋白口袋特征

        a = [k for k in range(len(compose_knn_edge_index[0])) if compose_knn_edge_index[0][k] in idx_protein and compose_knn_edge_index[1][k] in idx_protein]
        #edge_index是蛋白-蛋白边的索引
        edge_index= (compose_knn_edge_index[0][a],compose_knn_edge_index[1][a])

        compose_knn_edge_feature = compose_knn_edge_feature[a,:]
        #compose_pos长度和现在的边feature长度不一样 POS 1573
        h_compose = self.encoder(
            node_attr=h_compose,#后续传入idx_protein,只提取蛋白的
            # pos=compose_pos[idx_protein],protein最大下标为1917，但这里会使得超出长度，后面再分割idx_protein
            pos=compose_pos,#后续操作会因为edge_index舍去配体信息
            edge_index=edge_index,#只保留了蛋白的
            edge_feature=compose_knn_edge_feature,#只保留了蛋白的
            idx_protein=idx_protein,
        )
        max_group = torch.max(batch).item() + 1  # 获取最大组索引
        group_lengths = torch.bincount(batch)  # 计算每个组的长度
        max_length = torch.max(group_lengths).item()  # 获取最大组的长度
        #根据索引将原始张量划分为子张量并进行填充。
        h_tensor0=torch.zeros(max_group, max_length, h_compose[0].size(1))
        h_tensor1=torch.zeros(max_group, max_length, h_compose[1].size(1), h_compose[1].size(2))
        for i in range(max_group):
            group_indices = (batch == i).nonzero().squeeze(1)  # 获取当前组的索引
            group_data = h_compose[0][group_indices]  # 获取当前组的数据
            pad_length = max_length - len(group_data)  # 计算需要填充的长度
            padded_group_data = torch.nn.functional.pad(group_data, (0,0,0, pad_length), value=0)  # 进行填充
            h_tensor0[i]=padded_group_data
        for i in range(max_group):
            group_indices = (batch == i).nonzero().squeeze(1)  # 获取当前组的索引
            group_data = h_compose[1][group_indices]  # 获取当前组的数据
            pad_length = max_length - len(group_data)  # 计算需要填充的长度
            padded_group_data = torch.nn.functional.pad(group_data, (0, 0, 0, 0, 0, pad_length),value=0)  # 进行填充
            h_tensor1[i]=padded_group_data
        h_compose[0],h_compose[1]=h_tensor0,h_tensor1
        # print(h_compose[0].size())
        # print(h_compose[1].size())
        h_compose[0] = h_compose[0].to(device)
        h_compose[1] = h_compose[1].to(device)
        #把h_compose对应分成32个批次，然后加上pad，填充到最长的长度
        encoder_outputs = self.frontier_pred(
            h_compose,
        )

        print(encoder_outputs.size())
        encoder_outputs=encoder_outputs.to(device)
        decoder_outputs=self.decoder(smiles_index, encoder_outputs,tgt_len,prop)
        dec_logits = self.projection(decoder_outputs)
        if self.num_props:
            num = int(bool(self.num_props))
        else:
            num = 0
        dec_logits = dec_logits[:, num:, :]
        return dec_logits.reshape(-1, dec_logits.size(-1))

