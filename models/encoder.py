import torch
import torch.nn as nn

from models.module import PositionwiseFeedForward, LayerNorm, MultiHeadedAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout, attn, context_attn):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = attn
        self.context_attn = context_attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, graph_embed):
        input_norm = self.layer_norm(inputs)
        context, attn = self.self_attn(input_norm, input_norm, input_norm)

        query = self.dropout(context).squeeze(1) + inputs

        if graph_embed is not None:
            query_norm = self.layer_norm_2(query)
            mid, context_attn = self.context_attn(graph_embed, graph_embed, query_norm)

            out = self.dropout(mid).squeeze(1) + query
        else:
            out = query
            context_attn = None

        return self.feed_forward(out), context_attn


class TransformerEncoder(nn.Module):
    def __init__(self, config, attn_modules):
        super(TransformerEncoder, self).__init__()

        d_model = config.d_model
        self.num_layers = config.num_layers
        dropout = config.dropout

        self.attn_modules = attn_modules
        self.context_attn_modules = attn_modules

        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, d_model, dropout, self.attn_modules[i], self.context_attn_modules[i])
             for i in range(self.num_layers)])
        self.layer_norm = LayerNorm(d_model)

    def forward(self, out, graph_embed=None):
        '''
        :param src: [src_len, batch_size]
        :param bond: [batch_size, src_len, src_len, 7]
        :return:
        '''

        # Run the forward pass of every layer of the transformer.
        for i in range(self.num_layers):
            out, attn = self.transformer[i](out, graph_embed)

        out = self.layer_norm(out)
        return out
