import math

import torch
import torch.nn as nn


def init_params(module):
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class GenerateNodeFeature(nn.Module):
    """
    Compute node features for each node in the graph. 初始的节点嵌入加入度和出度的特征
    """
    def __init__(self, num_heads, num_users, num_in_degree, num_out_degree, hidden_dim):
        super(GenerateNodeFeature, self).__init__()
        self.num_heads = num_heads
        self.num_users = num_users
        self.in_degree_encoder = nn.Embedding(num_in_degree + 1, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree + 1, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)
        self.in_degree_encoder.weight.data.normal_(0, 0.02)
        self.out_degree_encoder.weight.data.normal_(0, 0.02)
        self.graph_token.weight.data.normal_(0, 0.02)

    def forward(self, batched_data):
        node_feature, in_degree, out_degree = (
            batched_data["features"],
            batched_data["in_degree"],  # batch_size * num_nodes
            batched_data["out_degree"],
        )
        n_graph, n_node = node_feature.size()[:2]
        node_feature = (
            node_feature  # batch_size * num_node * embedding
            + self.in_degree_encoder(in_degree)  # batch_size * num_node * embedding
            + self.out_degree_encoder(out_degree)
        )
        graph_token_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)  # batch_size * 1 * embedding
        graph_node_feature = torch.cat([graph_token_feature, node_feature], dim=1)
        return graph_node_feature  # batch_size * (num_node + 1) * embedding


class AttnBias(nn.Module):
    """
    Compute attention bias for each head.
    """
    def __init__(self, num_heads):
        super(AttnBias, self).__init__()
        self.num_heads = num_heads
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)
        self.apply(init_params)

    def forward(self, batched_data):  # timeline edge_strength
        timeline = batched_data["timeline"]  # batch_size * num_node
        edge_strength = batched_data["edge_strength"]  # batch_size * max_node * max_node
        x = batched_data["features"]  # batch_size * max_node * embedding
        n_graph, n_node = x.size()[:2]

        graph_attn_bias = torch.zeros(n_graph, self.num_heads, (n_node + 1), (n_node + 1), device='cuda')  # batch * (node + 1) * (node + 1)
        eye = torch.eye(timeline.size(1), device='cuda').unsqueeze(0)  # 1 * max_node * max_node
        diagonal = timeline.unsqueeze(-1) * eye  # batch_size * num_node * num_node
        diagonal = diagonal.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + diagonal  # batch_size * num_heads * num_node * num_node
        # 这里（上面）存在一些维度问题(改完了)
        # TODO: 降低空间复杂度或者减少节点数量（采样）
        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        edge_input = edge_strength.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input

        return graph_attn_bias  # batch_size * num_heads * num_nodes * num_nodes