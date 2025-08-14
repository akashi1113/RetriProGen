import torch
from torch.nn import Module
import torch.nn as nn
import torch.nn.functional as F

from .atten_module import MultiHeadedAttention
from .embedding import GVP
from .encoder import TransformerEncoder
from .interaction import get_interaction_vn
from .frontier import FrontierLayerVN
from .decoder import Decoder
from transformers import AutoTokenizer, AutoModel

from .search import BeamHypotheses


def get_attention_layer(num_layers, heads, d_model, dropout=0.0, device='cpu'):
    multihead_attn_modules = nn.ModuleList(
        [MultiHeadedAttention(heads, d_model, d_model, dropout=dropout)
         for _ in range(num_layers)])
    return multihead_attn_modules.to(device)


class MolT5EmbeddingModule(torch.nn.Module):
    def __init__(self, model_path):
        super(MolT5EmbeddingModule, self).__init__()
        self.path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModel.from_pretrained(self.path)

    def forward(self, smiles_sequences, device):
        embeddings = []
        for smiles_sequence in smiles_sequences:
            inputs = self.tokenizer(smiles_sequence, return_tensors="pt", padding=True, truncation=True)
            inputs = inputs.to(device)
            with torch.no_grad():
                self.model = self.model.to(device)
                outputs = self.model.encoder(**inputs)  # 仅使用编码器部分
                # outputs=outputs.to(device)
            embedding = outputs.last_hidden_state.mean(dim=1)  # 使用均值池化提取嵌入
            embeddings.append(embedding)
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.to(device)
        return embeddings


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


def get_crossattention_layer(config):
    heads = config.num_heads
    d_model = config.d_model
    num_layers = config.num_layers
    dropout = config.dropout

    ca_layer = nn.ModuleList(
        [MultiHeadedAttention(heads, d_model, d_model, dropout=dropout)
         for _ in range(num_layers)])
    return ca_layer


class TranMResProModel(Module):

    def __init__(self, config, protein_res_feature_dim, ligand_atom_feature_dim, num_props=None):
        super().__init__()
        self.config = config
        # 得到蛋白embedding，还需要进入encoder
        self.num_props = num_props
        self.emb_dim = [config.hidden_channels, config.hidden_channels_vec]
        self.protein_res_emb = GVP(protein_res_feature_dim, self.emb_dim)
        self.ligand_atom_emb = GVP(ligand_atom_feature_dim, self.emb_dim)
        # 编码器
        self.encoder = get_interaction_vn(config.encoder)
        # self.cross_encoder = get_crossattention_layer(config.cross_encoder)
        MultiHeadedAttention_layer = get_crossattention_layer(config.cross_encoder)
        self.cross_encoder = TransformerEncoder(config.cross_encoder, MultiHeadedAttention_layer)
        in_sca, in_vec = self.encoder.out_sca, self.encoder.out_vec
        self.frontier_pred = FrontierLayerVN(in_sca=in_sca, in_vec=in_vec,
                                             hidden_dim_sca=128, hidden_dim_vec=32)
        self.decoder = Decoder(config.decoder, self.num_props)

        self.projection = nn.Linear(config.hidden_channels, len(config.decoder.smiVoc), bias=False)
        self.projection2 = nn.Linear(512, 256, bias=False)
        self.projection3 = nn.Linear(256, 256, bias=False)

    def forward(self, batch, smiles_index, compose_feature, compose_vec, idx_ligand, idx_protein,
                compose_pos, compose_knn_edge_index, compose_knn_edge_feature, tgt_len, ret_embeddings, prop=None,
                device='cpu'):
        h_compose = embed_compose_GVP(compose_feature, compose_vec, idx_ligand, idx_protein,
                                      self.ligand_atom_emb, self.protein_res_emb, self.emb_dim)

        a = [k for k in range(len(compose_knn_edge_index[0])) if
             compose_knn_edge_index[0][k] in idx_protein and compose_knn_edge_index[1][k] in idx_protein]

        edge_index = (compose_knn_edge_index[0][a], compose_knn_edge_index[1][a])

        compose_knn_edge_feature = compose_knn_edge_feature[a, :]

        h_compose = self.encoder(
            node_attr=h_compose,
            pos=compose_pos,
            edge_index=edge_index,
            edge_feature=compose_knn_edge_feature,
            idx_protein=idx_protein,
        )

        # compsose_encoder_outputs = self.frontier_pred(h_compose)
        # compsose_embed_transformed = self.projection3(compsose_encoder_outputs)

        # # split protein feature and ligand feature
        protein_encoder_outputs = self.frontier_pred(h_compose, idx_protein)
        protein_embed_transformed = self.projection3(protein_encoder_outputs)
        # ligand_encoder_outputs = self.frontier_pred(h_compose, idx_ligand)

        # merge retmol features
        batch_size = torch.max(batch).item() + 1
        group_lengths = torch.bincount(batch)
        max_length = torch.max(group_lengths).item()
        h_tensor0 = torch.zeros(batch_size, max_length, protein_embed_transformed.size(1)).to(device)
        for i in range(batch_size):
            group_indices = (batch == i).nonzero().squeeze(1)  # 获取当前组的索引
            group_data = protein_embed_transformed[group_indices]  # 获取当前组的数据
            pad_length = max_length - len(group_data)  # 计算需要填充的长度
            padded_group_data = torch.nn.functional.pad(group_data, (0, 0, 0, pad_length), value=0)  # 进行填充
            h_tensor0[i] = padded_group_data

        if ret_embeddings is not None:
            ret_mol_embed_transformed = self.projection2(ret_embeddings)
            # fusion module
            h_tensor0 = self.cross_encoder(h_tensor0, ret_mol_embed_transformed)
            # for layer in self.cross_encoder:
            #     h_tensor0, atten = layer(ret_mol_embed_transformed, ret_mol_embed_transformed, h_tensor0)
            #     # TODO: add residual module
        decoder_outputs = self.decoder(smiles_index, h_tensor0, tgt_len, prop, device)
        dec_logits = self.projection(decoder_outputs)
        if self.num_props:
            num = int(bool(self.num_props))
        else:
            num = 0
        dec_logits = dec_logits[:, num:, :]
        return dec_logits.reshape(-1, dec_logits.size(-1))

    def gen(self, batch, compose_feature, compose_vec, idx_ligand, idx_protein, compose_pos, compose_knn_edge_index,
            compose_knn_edge_feature, max_length, ret_embeddings, smiVoc, num_beams, prop, topk,
            device='cpu'):

        batch_size = torch.max(batch).item() + 1

        cur_len = 1
        vocab_size = len(smiVoc)
        sos_token_id = smiVoc.index('&')
        eos_token_id = smiVoc.index('$')
        pad_token_id = smiVoc.index('^')
        beam_scores = torch.zeros((batch_size, num_beams)).to(device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        done = [False for _ in range(batch_size)]
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty=0.7)
            for _ in range(batch_size)
        ]

        input_ids = torch.full((batch_size * num_beams, 1), sos_token_id, dtype=torch.long).to(device)

        # encode the protein feature and last step
        h_compose = embed_compose_GVP(compose_feature, compose_vec, idx_ligand, idx_protein,
                                      self.ligand_atom_emb, self.protein_res_emb, self.emb_dim)
        a = [k for k in range(len(compose_knn_edge_index[0])) if
             compose_knn_edge_index[0][k] in idx_protein and compose_knn_edge_index[1][k] in idx_protein]
        edge_index = (compose_knn_edge_index[0][a], compose_knn_edge_index[1][a])
        compose_knn_edge_feature = compose_knn_edge_feature[a, :]
        h_compose = self.encoder(
            node_attr=h_compose,
            pos=compose_pos,
            edge_index=edge_index,
            edge_feature=compose_knn_edge_feature,
            idx_protein=idx_protein,
        )
        protein_encoder_outputs = self.frontier_pred(h_compose, idx_protein)
        protein_embed_transformed = self.projection3(protein_encoder_outputs)

        group_lengths = torch.bincount(batch)
        input_max_length = torch.max(group_lengths).item()
        encoder_out = torch.zeros(batch_size, input_max_length, protein_embed_transformed.size(1)).to(device)
        for i in range(batch_size):
            group_indices = (batch == i).nonzero().squeeze(1)  # 获取当前组的索引
            group_data = protein_embed_transformed[group_indices]  # 获取当前组的数据
            pad_length = input_max_length - len(group_data)  # 计算需要填充的长度
            padded_group_data = torch.nn.functional.pad(group_data, (0, 0, 0, pad_length), value=0)  # 进行填充
            encoder_out[i] = padded_group_data
        if ret_embeddings is not None:
            ret_mol_embed_transformed = self.projection2(ret_embeddings)
            # fusion module
            encoder_out = self.cross_encoder(encoder_out, ret_mol_embed_transformed)
        # if ret_embeddings is not None:
        #
        #     # merge retmol features
        #     ret_mol_embed_transformed = self.projection2(ret_embeddings)
        #     # fusion module
        #     for layer in self.cross_encoder:
        #         encoder_out, atten = layer(ret_mol_embed_transformed, ret_mol_embed_transformed, encoder_out)
        encoder_out = encoder_out.repeat_interleave(num_beams, 0)
        # encoder_out = encoder_out.repeat((num_beams, 1, 1))
        while cur_len < max_length:
            decoder_outputs = self.decoder(input_ids, encoder_out, cur_len, prop, device)
            dec_logits = self.projection(decoder_outputs)

            next_token_logits = dec_logits[:, -1, :]
            scores = F.log_softmax(next_token_logits, dim=-1)
            next_scores = scores + beam_scores[:, None].expand_as(scores)
            next_scores = next_scores.view(
                batch_size, num_beams * vocab_size
            )
            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            next_batch_beam = []

            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)
                    continue
                next_sent_beam = []
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                        zip(next_tokens[batch_idx], next_scores[batch_idx])
                ):
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size

                    effective_beam_id = batch_idx * num_beams + beam_id
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                    if len(next_sent_beam) == num_beams:
                        break

                    done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len
                    )

                next_batch_beam.extend(next_sent_beam)

            if all(done):
                break

            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])

            input_ids = input_ids[beam_idx, :]
            encoder_out = encoder_out[beam_idx, :]

            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
        output_num_return_sequences_per_batch = topk
        output_batch_size = output_num_return_sequences_per_batch * batch_size
        sent_lengths = input_ids.new(output_batch_size)
        best = []
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            decoded = torch.stack(best).type(torch.long)

        return decoded
