import torch
import torch.nn.functional as F


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty):
        self.max_length = max_length - 1
        self.length_penalty = length_penalty
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


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


def beam_search(model, smiVoc, num_beams, batch_size, max_length, topk, batch, batch_ret_embeddings, prop=None,
                device='cpu'):
    batch_protein = batch['batch']
    compose_feature = batch['compose_feature'].float()
    compose_vec = batch['compose_vec']
    idx_ligand = batch['idx_ligand_ctx_in_compose']
    idx_protein = batch['idx_protein_in_compose']
    compose_pos = batch['compose_pos']
    compose_knn_edge_index = batch['compose_knn_edge_index']
    compose_knn_edge_feature = batch['compose_knn_edge_feature']

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
                                  model.ligand_atom_emb, model.protein_res_emb, model.emb_dim)
    a = [k for k in range(len(compose_knn_edge_index[0])) if
         compose_knn_edge_index[0][k] in idx_protein and compose_knn_edge_index[1][k] in idx_protein]
    edge_index = (compose_knn_edge_index[0][a], compose_knn_edge_index[1][a])
    compose_knn_edge_feature = compose_knn_edge_feature[a, :]
    h_compose = model.encoder(
        node_attr=h_compose,
        pos=compose_pos,
        edge_index=edge_index,
        edge_feature=compose_knn_edge_feature,
        idx_protein=idx_protein,
    )
    protein_encoder_outputs = model.frontier_pred(h_compose, idx_protein)
    protein_embed_transformed = model.projection3(protein_encoder_outputs)

    ret_mol_embed_transformed = model.projection2(batch_ret_embeddings)

    # merge retmol features

    group_lengths = torch.bincount(batch_protein)
    input_max_length = torch.max(group_lengths).item()
    encoder_out = torch.zeros(batch_size, input_max_length, protein_embed_transformed.size(1)).to(device)
    for i in range(batch_size):
        group_indices = (batch_protein == i).nonzero().squeeze(1)  # 获取当前组的索引
        group_data = protein_embed_transformed[group_indices]  # 获取当前组的数据
        pad_length = input_max_length - len(group_data)  # 计算需要填充的长度
        padded_group_data = torch.nn.functional.pad(group_data, (0, 0, 0, pad_length), value=0)  # 进行填充
        encoder_out[i] = padded_group_data

    # fusion module
    #for layer in model.cross_encoder:
    #    encoder_out, atten = layer(ret_mol_embed_transformed, ret_mol_embed_transformed, encoder_out)
    encoder_out = model.cross_encoder(encoder_out, ret_mol_embed_transformed)
    encoder_out =encoder_out.repeat_interleave(num_beams,0)

    while cur_len < max_length:
        decoder_outputs = model.decoder(input_ids, encoder_out, cur_len, prop, device)
        dec_logits = model.projection(decoder_outputs)

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
