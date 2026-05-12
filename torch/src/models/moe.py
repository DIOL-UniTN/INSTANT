import torch
import torch.nn as nn
import torch.nn.functional as F
from .fff import FFF
from .ff import FF
from utils.act import TSoftmax

# Define the Gating Network class
class LinearGating(nn.Module):
    def __init__(self, in_features, num_experts, T):
        super(LinearGating, self).__init__()
        self.gate = nn.Linear(in_features, num_experts)
        self.softmax = TSoftmax.apply
        self.temp = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x):
        x = x.view(len(x), -1)
        return self.softmax(self.gate(x), self.temp)

# Define the Mixture of Experts Layer class
class MoE(nn.Module):
    def __init__(self, in_features, width, out_features, 
                 num_experts, soft_T: float):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(
                [FF(in_features, width, out_features) 
                 for _ in range(num_experts)]
                )
        self.gate = LinearGating(in_features, num_experts, T=soft_T)
        self.width = width
        self.num_experts = num_experts

    def forward(self, x, num_top_experts: int = 0):
        if not num_top_experts:
            num_top_experts = self.num_experts
        gating_scores = self.gate(x)
        top_gating_scores, top_indices = gating_scores.topk(num_top_experts, dim=1, sorted=False)
        # Create a mask to zero out the contributions of non-topk experts
        mask = torch.zeros_like(gating_scores).scatter_(1, top_indices, 1)
        # Use the mask to retain only the topk gating scores
        gating_scores = gating_scores * mask
        # Normalize the gating scores to sum to 1 across the selected top experts
        gating_scores = F.normalize(gating_scores, p=1, dim=1)
        
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.einsum('be,beo->bo', gating_scores, expert_outputs)
        return output

    def get_config(self):
        return {
                "width": self.width,
                "gate": "linear",
                "num_experts": self.num_experts,
                }

class MoEFFF(nn.Module):
    def __init__(self, in_features, leaf_width, out_features, depth, num_experts):
        super(MoEFFF, self).__init__()
        self.experts = nn.ModuleList([FFF(in_features, leaf_width, out_features, depth) for _ in range(num_experts)])
        self.gate = LinearGating(in_features, num_experts)
        self.leaf_width = leaf_width
        self.depth = depth
        self.num_experts = num_experts

    def forward(self, x, num_top_experts: int = 0, expert_alpha: float = 1.0):
        if num_top_experts != 1:
            num_top_experts = self.num_experts
        gating_scores = self.gate(x)
        top_gating_scores, top_indices = gating_scores.topk(num_top_experts, dim=1, sorted=False)
        # Create a mask to zero out the contributions of non-topk experts
        mask = torch.zeros_like(gating_scores).scatter_(1, top_indices, 1)
        # Use the mask to retain only the topk gating scores
        gating_scores = gating_scores * mask
        # Normalize the gating scores to sum to 1 across the selected top experts
        gating_scores = F.normalize(gating_scores, p=1, dim=1)
        
        expert_outputs = torch.stack([expert(x, a=expert_alpha)[0] for expert in self.experts], dim=1)
        output = torch.einsum('be,beo->bo', gating_scores, expert_outputs)
        return output

    def get_config(self):
        return {
                "leaf_width": self.leaf_width,
                "depth": self.depth,
                "gate": "linear",
                "num_experts": self.num_experts,
                }
