import torch
import torch.nn as nn


from graphormer_graph_encoder import GraphormerGraphEncoder
from parsers import create_parser
from SDEFunc import SDEFunc

parser = create_parser()
args = parser.parse_args()


class GraphormerNeuralSDEModel(nn.Module):
    def __init__(self, user_size, in_degree, out_degree):
        super(GraphormerNeuralSDEModel, self).__init__()
        self.embedding_layer = nn.Embedding(user_size, args.emb_dim, padding_idx=0)
        self.graphormergraphencoder = GraphormerGraphEncoder(user_size, in_degree, out_degree)
        self.sdefunc = SDEFunc(args.emb_dim, args.emb_dim)
        self.predictor = nn.Sequential(
            nn.Linear(args.emb_dim * 2, 1),
            nn.ReLU()
        )

    def forward(self, data):
        x = data
        batch_size = len(x)  # batch_size
        num_nodes = [x_i.x.size(0) for x_i in x]
        max_nodes = max(num_nodes)  # num_nodes in one graph
        node_features = torch.zeros(batch_size, max_nodes, args.emb_dim, device='cuda')  # batch_size(num_graph) * num_nodes * embed_dim
        node_timeline = torch.zeros(batch_size, max_nodes, device='cuda')  # batch_size * num_nodes
        attn_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device='cuda')
        in_degree = torch.zeros(batch_size, max_nodes, dtype=torch.long, device='cuda')  # batch_size * num_nodes
        out_degree = torch.zeros(batch_size, max_nodes, dtype=torch.long, device='cuda')
        edge_strength = torch.zeros(batch_size, max_nodes, max_nodes, device='cuda')
        for i in range(batch_size):
            node_features[i, :num_nodes[i], :] = self.embedding_layer(x[i].x.squeeze())  #
            node_timeline[i, :num_nodes[i]] = x[i].node_timestamp
            attn_mask[i, :num_nodes[i]] = True
            unique_nodes, counts = torch.unique(x[i].edge_index[1], return_counts=True)
            in_degree[i, unique_nodes] = counts
            non_zero_in_mask = in_degree > 0
            in_degree[non_zero_in_mask] = torch.floor(torch.log2(in_degree[non_zero_in_mask].float())).long()
            unique_nodes, counts = torch.unique(x[i].edge_index[0], return_counts=True)
            out_degree[i, unique_nodes] = counts
            non_zero_out_mask = out_degree > 0
            out_degree[non_zero_out_mask] = torch.floor(torch.log2(out_degree[non_zero_out_mask].float())).long()
            for j in range(len(x[i].edge_strength)):
                edge_strength[i, x[i].edge_index[0][j], x[i].edge_index[1][j]] = x[i].edge_strength[j]
        x = {
            'features': node_features,
            'attn_mask': attn_mask,
            'timeline': node_timeline,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'edge_strength': edge_strength,
            'num_nodes': num_nodes
        }
        x_graphormer_out, graph_rep = self.graphormergraphencoder(x)  # h:[(len + 1), batch, embedding]
        non_zero = (node_timeline != 0).sum(dim=1)   # [batch]
        mask = (torch.arange(x_graphormer_out[0].size()[0], device='cuda').unsqueeze(1) < non_zero.unsqueeze(0))
        mask = mask.unsqueeze(-1).expand(-1, -1, args.emb_dim)
        x_sde_input = x_graphormer_out[0] * mask  # [len+1, batch, embedding]
        x_sde_input = x_sde_input[1:, :, :]
        x_sde_output = self.sdefunc(x_sde_input, node_timeline, non_zero)  # [batch_size, hidden_dim]
        pred = self.predictor(torch.cat((graph_rep, x_sde_output), dim=-1))
        return pred
