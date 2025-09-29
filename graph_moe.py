import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HeteroConv, GATConv, global_mean_pool, BatchNorm, MessagePassing, SAGEConv
from torch_geometric.utils import softmax, add_self_loops, degree, k_hop_subgraph


class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, dropout=0.1):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_weights = self.softmax(gate_logits)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.bmm(gate_weights.unsqueeze(1), expert_outputs).squeeze(1)
        return output
    

class GGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, dropout=0.2):
        super(GGNNLayer, self).__init__(aggr='add')  
        self.edge_mlp = nn.Linear(edge_dim, in_channels)
        self.gru = nn.GRUCell(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        if edge_attr is not None:
            loop_attr = torch.zeros(x.size(0), edge_attr.size(1), device=edge_attr.device)  
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)  

        edge_attr = self.edge_mlp(edge_attr)
        m = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        x = self.gru(m, x)
        x = self.dropout(x)
        return x

    def message(self, x_j, edge_attr):
        return x_j + edge_attr


class GNNWithMoE(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_experts=4, dropout=0.2):
        super(GNNWithMoE, self).__init__()
        self.edge_types =  ["reg", "jump","return", "call"]

        self.edge_dim = len(self.edge_types)

        self.conv1 = GGNNLayer(num_node_features, hidden_dim, edge_dim=self.edge_dim, dropout=dropout)
        self.conv2 = GGNNLayer(hidden_dim, hidden_dim, edge_dim=self.edge_dim, dropout=dropout)

        self.attention_fc = nn.Linear(hidden_dim, 1)

        self.moe = MoE(hidden_dim, hidden_dim, num_experts, dropout=dropout)

        self.proj = nn.Linear(1, 100)

        self.fc = nn.Linear(hidden_dim + 100, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.prefix_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 69)
        )

    def forward(self, data):

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        static_prob = data.prob_dist

        ldxb_flag = data.ldxb_flag

        ldxb_flag = self.proj(ldxb_flag.unsqueeze(1))

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.moe(x)

        graph_emb = global_mean_pool(x, batch)

        graph_emb_with_prefix = torch.cat([ldxb_flag, graph_emb], dim=1)

        logits = self.fc(graph_emb_with_prefix)

        return graph_emb, graph_emb, logits, None # prefix

