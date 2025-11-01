import torch
import torch.nn as nn

from generate_node_feature import GenerateNodeFeature, AttnBias
from graphormer_graph_encoder_layer import GraphormerGraphEncoderLayer
from multihead_attention import MultiheadAttention
'''graphormer_graph_encoder.py 是图编码器的主文件，负责整体的编码流程。
graphormer_graph_encoder_layer.py 定义了每一层的结构，包括自注意力和前馈神经网络。 若干个这个层可以组成graphormer_graph_encoder。 对下面两个的组装
generate_node_feature.py 负责生成节点特征和注意力偏置。 图最直接的接口
multihead_attention.py 实现了多头注意力机制，是Transformer模型的核心组件。  第三个的框架'''


def init_graphormer_params(module):

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class GraphormerGraphEncoder(nn.Module):
    def __init__(
        self, user_size, num_in_degree, num_out_degree,
        num_encoder_layers: int = 16,  # 12个编码器相连？
        embedding_dim: int = 64,  # 有点多？改成64或128  原先是768
        ffn_embedding_dim: int = 128,  # 有点多？改成64或128 原先是768
        num_attention_heads: int = 8,  # 有点多？改成8或12或16 原先是32
        dropout: float = 0.1,  # 可以先留着，我自己nn.dropout
        attention_dropout: float = 0.1,  # 可以先留着，我自己nn.dropout 0.1
        activation_dropout: float = 0.1,  #可以先留着，我自己nn.dropout 0.1
        layerdrop: float = 0.0,  # 紧跟在计算特征后的dropout  0.0
        export: bool = False,
    ):

        super().__init__()
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.graph_node_feature = GenerateNodeFeature(
            num_heads=num_attention_heads,
            num_users=user_size,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim
        )

        self.graph_attn_bias = AttnBias(  # 空间和边  需要添加边的强度
            num_heads=num_attention_heads
        )
        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.dropout_module = nn.Dropout(self.layerdrop)
        '''if pre_layernorm:
            self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)'''
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_graphormer_graph_encoder_layer(  # 函数返回的是一个类layer 的对象 向函数传递参数
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,  #
                    attention_dropout=attention_dropout,  #
                    activation_dropout=activation_dropout,  #
                    export=export,
                    pre_layernorm=True,  # 可能会改
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        self.apply(init_graphormer_params)

    @staticmethod
    def build_graphormer_graph_encoder_layer(
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        export,
        pre_layernorm,
    ):
        return GraphormerGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            pre_layernorm=pre_layernorm,
        )

    def forward(self, batched_data, last_state_only=True):
        # compute padding mask. This is needed for multi-head attention
        data_x = batched_data["features"]  # batch_size * num_node * embedding
        n_graph, n_node = data_x.size()[:2]  # batch_size and num_node
        padding_mask = (data_x[:, :, 0]).eq(0)  # B x T  padding_mask 嵌入的第0维难道是索引吗，嵌入如果是填充的话，就填充0
        padding_mask_cls = torch.zeros(n_graph, 1, dtype=padding_mask.dtype, device='cuda')  # CLS 虚拟节点 batch_size * 1
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)  # batch_size * (num_node + 1) * 1

        x = self.graph_node_feature(batched_data)  # batch_size * (len + 1) * embedding

        attn_bias = self.graph_attn_bias(batched_data)  # batch_size * num_nodes * num_nodes * num_nodes

        x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)  # (len + 1) * batch * embedding

        inner_states = []
        if not last_state_only:
            inner_states.append(x)  # 初始状态

        for layer in self.layers:  # 对象layer，调用layer的forward()
            x = layer(
                x,  # (len + 1) * batch_size * embedding
                self_attn_padding_mask=padding_mask,  # 填充编码 batch_size * (num_node + 1) * 1
                self_attn_bias=attn_bias  # batch_size * num_nodes * num_nodes * num_nodes
            )
            if not last_state_only:
                inner_states.append(x)

        graph_rep = x[0, :, :]  # 图表示
        inner_states = [x]  # (len + 1) * batch * embedding
        return inner_states, graph_rep
