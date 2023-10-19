import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict
from itertools import combinations, product
import numpy as np

from arguments import Config


class LEU(nn.Module):
    __constants__ = ['alpha', 'inplace']
    alpha: float
    inplace: bool

    def __init__(self, alpha: float = 1.):
        super(LEU, self).__init__()
        self.alpha = alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # TODO more efficient
        output = torch.empty_like(input)
        output[input > 0] = self.alpha * torch.log(input[input > 0] + 1)
        output[input <= 0] = self.alpha * (torch.exp(input[input <= 0]) - 1)
        return output


class NoneAct(nn.Module):
    def forward(self, x):
        return x


class GELU(nn.Module):
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU_new(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def get_act(act_func):
    if isinstance(act_func, str):
        if act_func.lower() == 'relu':
            return nn.ReLU(inplace=True)
        elif act_func.lower() == 'tanh':
            return nn.Tanh()
        elif act_func.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif act_func.lower() == 'none':
            return NoneAct()
        elif act_func.lower() == 'elu':
            return nn.ELU()
        elif act_func.lower() == 'leu':
            return LEU()
        elif act_func.lower() == 'gelu':
            return GELU()
        elif act_func.lower() == 'gelu_new':
            return GELU_new()
        elif act_func.lower() == 'swish':
            return Swish()
        elif act_func.lower() == 'mish':
            return Mish()
        else:
            raise NotImplementedError
    else:
        return act_func


class Embeddings(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embedding = nn.Embedding(config.input_size, embedding_dim=config.embed_size)
        fan_in, fan_out = config.num_fields, config.embed_size
        # TODO: better initialization
        std = math.sqrt(2. / float(fan_in + fan_out))
        with torch.no_grad():
            init.normal_(self.embedding.weight, std=std)
        self.embed_norm = config.embed_norm
        if self.embed_norm:
            self.layer_norm = nn.LayerNorm(config.embed_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.embed_dropout_rate)

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        if self.embed_norm:
            embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class InnerProductLayer(nn.Module):
    def __init__(self, num_fields=None, output='product_sum'):
        super(InnerProductLayer, self).__init__()
        self.output_type = output
        if output not in ['product_sum', 'bi_interaction', 'inner_product', 'elementwise_product']:
            raise ValueError(f'InnerProductLayer output={output} is not supported')
        if num_fields is None:
            if output in ['inner_product', 'elementwise_product']:
                raise ValueError(f'num_fields is required when InnerProductLayer output={output}')
        else:
            p, q = zip(*list(combinations(range(num_fields), 2)))
            self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)
            self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)
            self.interaction_units = int(num_fields * (num_fields - 1) / 2)
            self.upper_triangle_mask = nn.Parameter(
                torch.triu(torch.ones(num_fields, num_fields), 1).type(torch.bool),
                requires_grad=False)

    def forward(self, feat_embed):
        if self.output_type in ['product_sum', 'bi_interaction']:
            sum_of_square = torch.sum(feat_embed, dim=1) ** 2  # sum then square
            square_of_sum = torch.sum(feat_embed ** 2, dim=1)  # square then sum
            bi_interaction = (sum_of_square - square_of_sum) * 0.5
            if self.output_type == 'bi_interaction':
                return bi_interaction
            else:
                return bi_interaction.sum(dim=-1, keepdim=True)
        elif self.output_type == 'inner_product':
            inner_product_matrix = torch.bmm(feat_embed, feat_embed.transpose(1, 2))
            flat_upper_triangle = torch.masked_select(inner_product_matrix, self.upper_triangle_mask)
            return flat_upper_triangle.view(-1, self.interaction_units)
        else:
            raise NotImplementedError


class OuterProductLayer(nn.Module):
    def __init__(self, num_fields, embed_size, kernel_type):
        super(OuterProductLayer, self).__init__()

        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_size, num_ix, embed_size
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_size
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise NotImplementedError
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, feat_embed):
        num_fields = feat_embed.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = feat_embed[:, row], feat_embed[:, col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)
            outer_output = torch.sum(kp * q, -1)
        else:
            outer_output = torch.sum(p * q * self.kernel.unsqueeze(0), -1)

        return outer_output


class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_size=128, num_hidden_layers=3,
                 hidden_act='relu', hidden_dropout_rate=0.5, batch_norm=False):
        super(MLPBlock, self).__init__()
        dense_layers = []
        for i in range(num_hidden_layers):
            dense_layers.append(nn.Linear(input_dim, hidden_size))
            if batch_norm:
                pass
            dense_layers.append(get_act(hidden_act))
            dense_layers.append(nn.Dropout(p=hidden_dropout_rate))
            input_dim = hidden_size
        self.dnn = nn.Sequential(*dense_layers)

    def forward(self, inputs):
        return self.dnn(inputs)


class CrossNetV2(nn.Module):
    def __init__(self, input_dim, num_cross_layers):
        super(CrossNetV2, self).__init__()
        self.num_layers = num_cross_layers
        self.cross_layers = nn.ModuleList(nn.Linear(input_dim, input_dim) for _ in range(num_cross_layers))

    def forward(self, X0):
        Xi = X0
        for i in range(self.num_layers):
            Xi = Xi + X0 * self.cross_layers[i](Xi)
        return Xi


class FGCNNBlock(nn.Module):
    def __init__(self,
                 num_fields,
                 embedding_dim,
                 channels=[3],
                 kernel_heights=[3],
                 pooling_sizes=[2],
                 recombined_channels=[2],
                 activation="Tanh",
                 batch_norm=True):
        super(FGCNNBlock, self).__init__()
        self.embedding_dim = embedding_dim
        conv_list = []
        recombine_list = []
        self.channels = [1] + channels
        input_height = num_fields
        for i in range(1, len(self.channels)):
            in_channel = self.channels[i - 1]
            out_channel = self.channels[i]
            kernel_height = kernel_heights[i - 1]
            pooling_size = pooling_sizes[i - 1]
            recombined_channel = recombined_channels[i - 1]
            conv_layer = [nn.Conv2d(in_channel, out_channel,
                                    kernel_size=(kernel_height, 1),
                                    padding=(int((kernel_height - 1) / 2), 0))] \
                         + ([nn.BatchNorm2d(out_channel)] if batch_norm else []) \
                         + [get_act(activation),
                            nn.MaxPool2d((pooling_size, 1), padding=(input_height % pooling_size, 0))]
            conv_list.append(nn.Sequential(*conv_layer))
            input_height = int(math.ceil(input_height / pooling_size))
            input_dim = input_height * embedding_dim * out_channel
            output_dim = input_height * embedding_dim * recombined_channel
            recombine_layer = nn.Sequential(nn.Linear(input_dim, output_dim),
                                            get_act(activation))
            recombine_list.append(recombine_layer)
        self.conv_layers = nn.ModuleList(conv_list)
        self.recombine_layers = nn.ModuleList(recombine_list)

    def forward(self, X):
        conv_out = X
        new_feature_list = []
        for i in range(len(self.channels) - 1):
            conv_out = self.conv_layers[i](conv_out)
            flatten_out = torch.flatten(conv_out, start_dim=1)
            recombine_out = self.recombine_layers[i](flatten_out)
            new_feature_list.append(recombine_out.reshape(X.size(0), -1, self.embedding_dim))
        new_feature_emb = torch.cat(new_feature_list, dim=1)
        return new_feature_emb


class SqueezeExtractionLayer(nn.Module):
    def __init__(self, config: Config):
        super(SqueezeExtractionLayer, self).__init__()
        reduced_size = max(1, int(config.num_fields / config.reduction_ratio))
        self.excitation = nn.Sequential(
            nn.Linear(config.num_fields, reduced_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_size, config.num_fields, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, feature_emb):
        Z = torch.mean(feature_emb, dim=-1, out=None)
        A = self.excitation(Z)
        V = feature_emb * A.unsqueeze(-1)
        return V


class BilinearInteractionLayer(nn.Module):
    def __init__(self, config: Config):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = config.bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(config.embed_size, config.embed_size, bias=False)
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(config.embed_size, config.embed_size, bias=False)
                                                 for i in range(config.num_fields)])
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(config.embed_size, config.embed_size, bias=False)
                                                 for i, j in combinations(range(config.num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i) * v_j
                             for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]
                             for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1]
                             for i, v in enumerate(combinations(feature_emb_list, 2))]
        return torch.cat(bilinear_list, dim=1)


class GraphLayer(nn.Module):
    def __init__(self, config: Config):
        super(GraphLayer, self).__init__()
        self.W_in = torch.nn.Parameter(torch.Tensor(config.num_fields, config.embed_size, config.embed_size))
        self.W_out = torch.nn.Parameter(torch.Tensor(config.num_fields, config.embed_size, config.embed_size))
        nn.init.xavier_normal_(self.W_in)
        nn.init.xavier_normal_(self.W_out)
        self.bias_p = nn.Parameter(torch.zeros(config.embed_size))

    def forward(self, g, h):
        h_out = torch.matmul(self.W_out, h.unsqueeze(-1)).squeeze(-1)  # broadcast multiply
        aggr = torch.bmm(g, h_out)
        a = torch.matmul(self.W_in, aggr.unsqueeze(-1)).squeeze(-1) + self.bias_p
        return a


class FiGNNBlock(nn.Module):
    def __init__(self, config: Config):
        super(FiGNNBlock, self).__init__()
        self.num_fields = config.num_fields
        self.embedding_dim = config.embed_size
        self.gnn_layers = config.num_hidden_layers
        self.use_residual = config.res_conn
        self.reuse_graph_layer = config.reuse_graph_layer
        if self.reuse_graph_layer:
            self.gnn = GraphLayer(config)
        else:
            self.gnn = nn.ModuleList([GraphLayer(config) for _ in range(self.gnn_layers)])
        self.gru = nn.GRUCell(config.embed_size, config.embed_size)
        self.src_nodes, self.dst_nodes = zip(*list(product(range(config.num_fields), repeat=2)))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.W_attn = nn.Linear(config.embed_size * 2, 1, bias=False)

    def build_graph_with_attention(self, feat_embed):
        src_emb = feat_embed[:, self.src_nodes, :]
        dst_emb = feat_embed[:, self.dst_nodes, :]
        concat_emb = torch.cat([src_emb, dst_emb], dim=-1)
        alpha = self.leaky_relu(self.W_attn(concat_emb))
        alpha = alpha.view(-1, self.num_fields, self.num_fields)
        try:
            device = feat_embed.get_device()
            mask = torch.eye(self.num_fields).to(device)
        except RuntimeError:
            mask = torch.eye(self.num_fields)
        alpha = alpha.masked_fill(mask.bool(), float('-inf'))
        graph = F.softmax(alpha, dim=-1)  # batch x field x field without self-loops
        return graph

    def forward(self, feat_embed):
        g = self.build_graph_with_attention(feat_embed)
        h = feat_embed
        for i in range(self.gnn_layers):
            if self.reuse_graph_layer:
                a = self.gnn(g, h)
            else:
                a = self.gnn[i](g, h)
            if self.gru is not None:
                a = a.view(-1, self.embedding_dim)
                h = h.view(-1, self.embedding_dim)
                h = self.gru(a, h)
                h = h.view(-1, self.num_fields, self.embedding_dim)
            else:
                h = a + h
            if self.use_residual:
                h += feat_embed
        return h


class AttentionalPrediction(nn.Module):
    def __init__(self, config: Config):
        super(AttentionalPrediction, self).__init__()
        self.linear1 = nn.Linear(config.embed_size, 1, bias=False)
        self.linear2 = nn.Sequential(nn.Linear(config.num_fields * config.embed_size, config.num_fields, bias=False),
                                     nn.Sigmoid())

    def forward(self, h):
        score = self.linear1(h).squeeze(-1)
        weight = self.linear2(h.flatten(start_dim=1))
        logits = (weight * score).sum(dim=1).unsqueeze(-1)
        return logits


class SelfAttention(nn.Module):
    def __init__(self, config: Config, output_attention_probs=False):
        super(SelfAttention, self).__init__()
        self.num_attn_heads = config.num_attn_heads
        self.attn_head_size = int(config.hidden_size / config.num_attn_heads)
        self.all_head_size = self.num_attn_heads * self.attn_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attn_probs_dropout_rate)
        self.output_attn_probs = output_attention_probs

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attn_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # B * H * N * N
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attn_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores * torch.ge(attention_mask, 0).float() + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # B * H * N * E
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attn_probs else (context_layer,)
        return outputs


class Attention(nn.Module):
    def __init__(self, config: Config, output_attention_probs=False):
        super(Attention, self).__init__()
        self.num_attn_heads = config.num_attn_heads
        self.attn_head_size = int(config.hidden_size / config.num_attn_heads)
        self.all_head_size = self.num_attn_heads * self.attn_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attn_probs_dropout_rate)
        self.output_attn_probs = output_attention_probs

    def transpose_queries(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attn_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_keys_values(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attn_heads, self.attn_head_size)
        x = x.view(*new_x_shape)
        # B * H * N * N * e
        return x.permute(0, 3, 1, 2, 4)

    def forward(self, query_states, key_states, attention_mask=None):
        """
        :param query_states: B * N * E
        :param key_states: B * N * N * E
        :param attention_mask:
        :return:
        """
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(key_states)
        mixed_value_layer = self.value(key_states)

        # B * H * N * e
        query_layer = self.transpose_queries(mixed_query_layer)
        # B * H * N * N * e
        key_layer = self.transpose_keys_values(mixed_key_layer)
        value_layer = self.transpose_keys_values(mixed_value_layer)

        # B * H * N * N
        # print(key_layer.shape, query_layer.shape)
        attention_scores = torch.matmul(key_layer, query_layer.unsqueeze(-1)).squeeze(-1)
        # print(attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attn_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores * torch.ge(attention_mask, 0).float() + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        # B * H * N * E
        # print(value_layer.shape, attention_probs.shape)
        context_layer = torch.matmul(value_layer.transpose(-1, -2), attention_probs.unsqueeze(-1)).squeeze(-1)
        # print(context_layer.shape)
        # exit(0)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attn_probs else (context_layer,)
        return outputs


class ProductLayer(nn.Module):
    def __init__(self, config: Config, c_in=1, c_out=1):
        super(ProductLayer, self).__init__()
        self.kernel = nn.Parameter(torch.empty(c_out, c_in, config.num_fields, config.hidden_size, config.hidden_size,
                                               dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(config.num_fields, c_out, dtype=torch.float32))
        nn.init.xavier_normal_(self.kernel.data, gain=math.sqrt(config.num_fields))
        nn.init.zeros_(self.bias)
        # TODO identity init
        # fan_in = c_in * embed_size * embed_size
        # fan_out = c_out * embed_size * embed_size
        # self.kernel.data.copy_(torch.eye(embed_size, dtype=torch.float32) * math.sqrt(2 / (fan_in + fan_out)))
        # self.bias.data.copy_(torch.zeros(c_out, dtype=torch.float32))

        self.agg_type = config.agg_type
        self.res_conn = config.res_conn
        if self.agg_type == 'attn':
            self.self_attn = SelfAttention(config, output_attention_probs=True)
        if config.prod_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm_first = config.norm_first
        else:
            self.layer_norm = None

    def forward(self, feat_embeds: torch.Tensor):
        """
        :param feat_embeds: B * N * c_in * E
        :return: B * N * c_out * E
        """
        c_out, c_in, N, E, _ = self.kernel.shape

        # p: B * N * c_in * E, k: c_out * c_in * N * E * E
        # p: (N * c_in) * B * E @ k: (N * c_in) * E * (c_out * E)
        # -> N * c_in * B * c_out * E
        # -> (B * N * E) * c_out * c_in
        pk = torch.bmm(feat_embeds.permute((1, 2, 0, 3)).reshape((N * c_in, -1, E)),
                       self.kernel.permute((2, 1, 3, 0, 4)).reshape((N * c_in, E, c_out * E))). \
            reshape((N, c_in, -1, c_out, E)).permute((2, 0, 4, 3, 1)).reshape((-1, c_out, c_in))

        if self.agg_type == 'attn':
            if self.layer_norm is not None and self.norm_first:
                q = self.layer_norm(feat_embeds)
            else:
                q = feat_embeds
            # (B * c_in) * N * E
            q = q.permute((0, 2, 1, 3)).reshape((-1, N, E))
            # (B * c_in) * N * E
            q, attn_weights = self.self_attn(q)
            # (B * N * E) * c_in * 1
            q = q.reshape((-1, c_in, N, E)).permute((0, 2, 3, 1)).reshape((-1, c_in, 1))

            # (B * N * E) * c_out -> B * N * c_out * E
            pkq = torch.bmm(pk, q).reshape((-1, N, E, c_out)).permute((0, 1, 3, 2))
        else:
            # B * N * E * c_out * c_in -> B * N * c_out * c_in * E
            pk = pk.reshape((-1, N, E, c_out, c_in)).permute((0, 1, 3, 4, 2))

            if self.layer_norm is not None and self.norm_first:
                q = self.layer_norm(feat_embeds)
            else:
                q = feat_embeds

            if self.agg_type == 'sum':
                q = q.sum(dim=1, keepdim=True).unsqueeze(2)
            else:
                # B * 1 * c_in * E -> B * 1 * 1 * c_in * E
                q = q.mean(dim=1, keepdim=True).unsqueeze(2)

            # B * N * c_out * E
            pkq = (pk * q).sum(dim=3)

        # N * c_out -> 1 * N * c_out * E
        pkq += self.bias.unsqueeze(0).unsqueeze(-1)

        if self.res_conn:
            if c_in == c_out or c_in == 1:
                pkq += feat_embeds
            else:
                assert c_in == c_out, 'res connection requires c_in == c_out or c_in == 1'

        if self.layer_norm is not None and not self.norm_first:
            pkq = self.layer_norm(pkq)

        return pkq


class IntermediateLayer(nn.Module):
    def __init__(self, config: Config):
        super(IntermediateLayer, self).__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.inter_act_func = get_act(config.hidden_act)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.res_conn = config.res_conn
        if config.inter_layer_norm:
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.norm_first = config.norm_first
        else:
            self.layer_norm = None
        self.dropout = nn.Dropout(config.hidden_dropout_rate)

    def forward(self, hidden_states):
        """
        :param hidden_states: B * N * C * E or B * N * E
        :return: same shape as hidden_states
        """
        input_tensor = hidden_states
        if self.layer_norm is not None and self.norm_first:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.inter_act_func(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.res_conn:
            hidden_states = hidden_states + input_tensor
        if self.layer_norm is not None and not self.norm_first:
            hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class MultiChannelOutputHead(nn.Module):
    def __init__(self, config: Config):
        super(MultiChannelOutputHead, self).__init__()
        self.config = config

        if 'fc' not in config.output_reduction:
            assert config.output_dim == 1, f'output_reduction = {config.output_reduction} requires output_dim = 1'

        self.output_reduction = config.output_reduction.split(',')
        
        if len(self.output_reduction) == 1 and self.output_reduction[0] == 'fc':
            self.fc_out = nn.Linear(config.num_fields * config.num_channels * config.embed_size, config.output_dim)
        elif len(self.output_reduction) == 2 and self.output_reduction[0] == 'mean' and self.output_reduction[1] == 'fc':
            self.fc_out = nn.Linear(config.num_channels * config.embed_size, config.output_dim)
        else:
            def get_pooler(pooler_type):
                if pooler_type == 'max':
                    return nn.MaxPool1d
                elif pooler_type == 'avg':
                    return nn.AvgPool1d
                else:
                    raise NotImplementedError

            self.pooler1 = get_pooler(self.output_reduction[2])(kernel_size=config.hidden_size) if \
                self.output_reduction[2] in ['max', 'avg'] else None
            self.pooler2 = get_pooler(self.output_reduction[1])(kernel_size=config.num_channels) if \
                self.output_reduction[1] in ['max', 'avg'] else None
            self.pooler3 = get_pooler(self.output_reduction[0])(kernel_size=config.num_fields) if \
                self.output_reduction[0] in ['max', 'avg'] else None

            out_dim = (config.embed_size if self.output_reduction[2] == 'fc' else 1) * \
                    (config.num_channels if self.output_reduction[1] == 'fc' else 1) * \
                    (config.num_fields if self.output_reduction[0] == 'fc' else 1)

            self.fc_out = nn.Linear(out_dim, config.output_dim) if out_dim > 1 else None

    def forward(self, hidden_states):
        """
        :param hidden_states: B * N * C * E
        :return:
        """
        if len(self.output_reduction) == 1 and self.output_reduction[0] == 'fc':
            output = self.fc_out(hidden_states.flatten(start_dim=1))
        elif len(self.output_reduction) == 2 and self.output_reduction[0] == 'mean' and self.output_reduction[1] == 'fc':
            hidden_states = torch.sum(hidden_states, dim=1) / self.config.num_fields
            output = self.fc_out(hidden_states.flatten(start_dim=1))
        else:
            _, N, C, E = hidden_states.shape
            if self.output_reduction[2] == 'sum':
                hidden_states = hidden_states.sum(dim=3)
            elif self.pooler1 is not None:
                if hidden_states.ndim < 4:
                    hidden_states = self.pooler1(hidden_states).squeeze(dim=-1)
                else:
                    hidden_states = self.pooler1(hidden_states.reshape(-1, N * C, E)). \
                        squeeze(dim=-1).reshape(-1, N, C)

            if self.output_reduction[1] == 'sum':
                hidden_states = hidden_states.sum(dim=2)
            elif self.pooler2 is not None:
                if hidden_states.ndim < 4:
                    hidden_states = self.pooler2(hidden_states).squeeze(dim=-1)
                else:
                    hidden_states = self.pooler2(hidden_states.permute(0, 1, 3, 2).reshape(-1, N * E, C)). \
                        squeeze(dim=-1).reshape(-1, N, E)

            if self.output_reduction[0] == 'sum':
                hidden_states = hidden_states.sum(dim=1)
            elif self.pooler3 is not None:
                if hidden_states.ndim < 4:
                    hidden_states = self.pooler3(hidden_states).squeeze(dim=-1)
                else:
                    hidden_states = not self.pooler3(hidden_states.permute(0, 2, 3, 1).reshape(-1, C * E, N)). \
                        squeeze(dim=-1).reshape(-1, C, E)

            output = hidden_states
            if self.fc_out is not None:
                output = self.fc_out(hidden_states.flatten(start_dim=1))

        return output


class CIN(nn.Module):
    def __init__(self, num_fields, cin_layer_units):
        super(CIN, self).__init__()
        self.cin_layer_units = cin_layer_units
        self.cin_layer = nn.ModuleDict()
        for i, unit in enumerate(self.cin_layer_units):
            in_channels = num_fields * self.cin_layer_units[i - 1] if i > 0 else num_fields ** 2
            out_channels = unit
            self.cin_layer["layer_" + str(i + 1)] = nn.Conv1d(in_channels,
                                                              out_channels,  # how many filters
                                                              kernel_size=1) # kernel output shape

    def forward(self, X_0):
        pooling_outputs = []
        batch_size = X_0.shape[0]
        embedding_dim = X_0.shape[-1]
        X_i = X_0
        for i in range(len(self.cin_layer_units)):
            hadamard_tensor = torch.einsum("bhd,bmd->bhmd", X_0, X_i)
            hadamard_tensor = hadamard_tensor.view(batch_size, -1, embedding_dim)
            X_i = self.cin_layer["layer_" + str(i + 1)](hadamard_tensor) \
                      .view(batch_size, -1, embedding_dim)
            pooling_outputs.append(X_i.sum(dim=-1))
        concate_vec = torch.cat(pooling_outputs, dim=-1)
        
        return concate_vec


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    def __init__(self, dropout_rate=0.):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = None
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, W_q, W_k, W_v, scale=None, mask=None):
        attention = torch.bmm(W_q, W_k.transpose(1, 2))
        if scale:
            attention = attention / scale
        if mask:
            attention = attention.masked_fill_(mask, -np.inf)
        attention = self.softmax(attention)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention, W_v)
        return output, attention
        

class InterHAt_AttentionalAggregation(nn.Module):
    '''
    agg attention for InterHAt
    '''
    def __init__(self, embedding_dim, hidden_dim=None):
        super(InterHAt_AttentionalAggregation, self).__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embedding_dim
        self.agg = nn.Sequential(nn.Linear(embedding_dim, hidden_dim), 
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, 1, bias=False),
                                 nn.Softmax(dim=1))

    def forward(self, X):
        # X: b x f x emb
        attentions = self.agg(X) # b x f x 1
        attention_out = (attentions * X).sum(dim=1) # b x emb
        return attention_out


class InterHAt_MultiHeadAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False):
        super(InterHAt_MultiHeadAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim // num_heads
        self.attention_dim = attention_dim
        self.output_dim = num_heads * attention_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.scale = attention_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, self.output_dim, bias=False)
        if input_dim != self.output_dim:
            self.W_res = nn.Linear(self.output_dim, input_dim, bias=False)
        else:
            self.W_res = None
        self.dot_product_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, query, key, value, mask=None):
        residual = query
        
        # linear projection
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size * self.num_heads, -1, self.attention_dim)
        key = key.view(batch_size * self.num_heads, -1, self.attention_dim)
        value = value.view(batch_size * self.num_heads, -1, self.attention_dim)
        if mask:
            mask = mask.repeat(self.num_heads, 1, 1)
        # scaled dot product attention
        output, attention = self.dot_product_attention(query, key, value, self.scale, mask)
        # concat heads
        output = output.view(batch_size, -1, self.output_dim)
        # final linear projection
        if self.W_res is not None:
            output = self.W_res(output)
        output = output.relu()
        if self.dropout is not None:
            output = self.dropout(output)
        if self.use_residual:
            output = output + residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        return output, attention


class InterHAt_MultiHeadSelfAttention(InterHAt_MultiHeadAttention):
    def forward(self, X):
        output, attention = super(InterHAt_MultiHeadSelfAttention, self).forward(X, X, X)
        return output


class InterHAt_FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, layer_norm=True, use_residual=True):
        super(InterHAt_FeedForwardNetwork, self).__init__()
        self.use_residual = use_residual
        if hidden_dim is None:
            hidden_dim = 4 * input_dim
        self.ffn = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, input_dim))
        self.layer_norm = nn.LayerNorm(input_dim) if layer_norm else None

    def forward(self, X):
        output = self.ffn(X)
        if self.use_residual:
            output += X
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        return output


class MultiHeadAttention(nn.Module):
    """ Multi-head attention module """

    def __init__(self, input_dim, attention_dim=None, num_heads=1, dropout_rate=0., 
                 use_residual=True, use_scale=False, layer_norm=False, align_to="input"):
        super(MultiHeadAttention, self).__init__()
        if attention_dim is None:
            attention_dim = input_dim // num_heads
        self.attention_dim = attention_dim
        self.output_dim = num_heads * attention_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.align_to = align_to
        self.scale = attention_dim ** 0.5 if use_scale else None
        self.W_q = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, self.output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, self.output_dim, bias=False)
        if input_dim != self.output_dim:
            if align_to == "output":
                self.W_res = nn.Linear(input_dim, self.output_dim, bias=False)
            elif align_to == "input":
                self.W_res = nn.Linear(self.output_dim, input_dim, bias=False)
        else:
            self.W_res = None
        self.dot_product_attention = ScaledDotProductAttention(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, query, key, value, mask=None):
        residual = query
        
        # linear projection
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)
        
        # split by heads
        batch_size = query.size(0)
        query = query.view(batch_size * self.num_heads, -1, self.attention_dim)
        key = key.view(batch_size * self.num_heads, -1, self.attention_dim)
        value = value.view(batch_size * self.num_heads, -1, self.attention_dim)
        if mask:
            mask = mask.repeat(self.num_heads, 1, 1)
        # scaled dot product attention
        output, attention = self.dot_product_attention(query, key, value, self.scale, mask)
        # concat heads
        output = output.view(batch_size, -1, self.output_dim)
        # final linear projection
        if self.W_res is not None:
            if self.align_to == "output": # AutoInt style
                residual = self.W_res(residual)
            elif self.align_to == "input": # Transformer stype
                output = self.W_res(output)
        if self.dropout is not None:
            output = self.dropout(output)
        if self.use_residual:
            output = output + residual
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = output.relu()
        return output, attention


class MultiHeadSelfAttention(MultiHeadAttention):
    def forward(self, X):
        output, attention = super(MultiHeadSelfAttention, self).forward(X, X, X)
        return output
        