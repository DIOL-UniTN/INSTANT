import math
import torch 
from torch import nn
from typing import Optional

from torchvision.transforms import Resize
from utils.fff import ASigmoid

class SigmoidFFF(nn.Module):
    def __init__(self, in_features: int, leaf_width: int, out_features: int, depth: int):
        super().__init__()

        self.in_features = in_features
        self.leaf_width = leaf_width
        self.out_features = out_features
        self.activation = nn.ReLU()
        self.node_activation = ASigmoid.apply

        if depth < 0 or in_features <= 0 or leaf_width <= 0 or out_features <= 0:
            raise ValueError("input/leaf/output widths and depth must be all positive integers")

        self.n_leaves = 2 ** depth
        self.n_nodes = 2 ** depth - 1
        self.depth = nn.Parameter(torch.tensor(depth, dtype=torch.long), requires_grad=False)

        self.n_nodeacts = 3
        self.alphas = nn.Parameter(torch.tensor([1, 10, 100]), requires_grad=False)
        self.node_acts = nn.Parameter(torch.ones((self.n_nodes, self.n_nodeacts), dtype=torch.float) / 3.0, requires_grad=True)

        l1_init_factor = 1.0 / math.sqrt(in_features)
        self.node_weights = nn.Parameter(torch.empty((self.n_nodes, in_features), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.node_biases = nn.Parameter(torch.empty((self.n_nodes, 1), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)

        l2_init_factor = 1.0 / math.sqrt(self.leaf_width)
        self.w1s = nn.Parameter(torch.empty((self.n_leaves, in_features, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.b1s = nn.Parameter(torch.empty((self.n_leaves, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w2s = nn.Parameter(torch.empty((self.n_leaves, leaf_width, out_features), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.b2s = nn.Parameter(torch.empty((self.n_leaves, out_features), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)

    def train_forward(self, x: torch.Tensor, a: float):
        # x has shape (batch_size, in_features)
        x = x.view(len(x), -1)
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        batch_size = x.shape[0]

        current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
        entropies = torch.zeros((batch_size, self.n_nodes), dtype=torch.float, device=x.device)
        sample_entropies = torch.zeros(self.n_nodes, dtype=torch.float, device=x.device)
        for current_depth in range(self.depth.item()):
            platform = torch.tensor(2 ** current_depth - 1, dtype=torch.long, device=x.device)
            next_platform = torch.tensor(2 ** (current_depth+1) - 1, dtype=torch.long, device=x.device)

            n_nodes = 2 ** current_depth
            current_weights = self.node_weights[platform:next_platform] # (n_nodes, in_features)    
            current_biases = self.node_biases[platform:next_platform]   # (n_nodes, 1)
            cur_node_acts = self.node_acts[platform:next_platform]

            boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(0, 1))      # (batch_size, n_nodes)
            boundary_plane_logits = boundary_plane_coeff_scores + current_biases.transpose(0, 1)# (batch_size, n_nodes)
            boundary_effect = torch.zeros_like(boundary_plane_logits)
            for j in range(self.n_nodeacts):
                boundary_effect += (self.node_activation(boundary_plane_logits, 
                                                               self.alphas[j]) * 
                                                cur_node_acts[:, j])
            # print(boundary_plane_logits.shape)
            # print(boundary_effect.shape)
            # print(boundary_effect.sum())
            # breakpoint()
            # boundary_effect2 = (self.node_activation(boundary_plane_logits, 
            #                                          self.alphas.repeat(n_nodes, 1)) * 
            #                     cur_node_acts)
            # print(boundary_effect2.sum(dim=1).shape)
            # print(boundary_effect2.sum())

            not_boundary_effect = 1 - boundary_effect                                   # (batch_size, n_nodes)

            platform_entropies = compute_entropy_safe(
                boundary_effect, not_boundary_effect
            ) # (batch_size, n_nodes)
            entropies[:, platform:next_platform] = platform_entropies   # (batch_size, n_nodes)
            boundary_prob =  boundary_effect.sum(dim=0) / batch_size
            sample_entropies[platform:next_platform] = (-boundary_prob * torch.log2(boundary_prob.sum(dim=0) + 1e-6) +
                                                        -(1 - boundary_prob) * torch.log2(1 - boundary_prob + 1e-6)
                                                        )
                
            mixture_modifier = torch.cat( # this cat-fu is to interleavingly combine the two tensors
                (not_boundary_effect.unsqueeze(-1), boundary_effect.unsqueeze(-1)),
                dim=-1
            ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)                                               # (batch_size, n_nodes*2, 1)
            current_mixture = current_mixture.view(batch_size, 2 * n_nodes, self.n_leaves // (2 * n_nodes)) # (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
            current_mixture.mul_(mixture_modifier)                                                          # (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
            current_mixture = current_mixture.flatten(start_dim=1, end_dim=2)                               # (batch_size, self.n_leaves)

            del mixture_modifier, boundary_effect, not_boundary_effect, boundary_plane_logits, boundary_plane_coeff_scores, current_weights, current_biases

        element_logits = torch.matmul(x, self.w1s.transpose(0, 1).flatten(1, 2))            # (batch_size, self.n_leaves * self.leaf_width)
        element_logits = element_logits.view(batch_size, self.n_leaves, self.leaf_width)    # (batch_size, self.n_leaves, self.leaf_width)
        element_logits += self.b1s.view(1, *self.b1s.shape)                                 # (batch_size, self.n_leaves, self.leaf_width)
        element_activations = self.activation(element_logits)                               # (batch_size, self.n_leaves, self.leaf_width)
        new_logits = torch.empty((batch_size, self.n_leaves, self.out_features), dtype=torch.float, device=x.device)
        for l in range(self.n_leaves):
            new_logits[:, l] = torch.matmul(
                element_activations[:, l],
                self.w2s[l]
                ) + self.b2s[l]
        # new_logits has shape (batch_size, self.n_leaves, self.out_features)

        # breakpoint()
        # breakpoint()
        # out_entropies = (-current_mixture * torch.log2(current_mixture + 1e-6) 
        #                  / math.log2(self.n_leaves)).sum(dim=1)
        new_logits *= current_mixture.unsqueeze(-1)
        final_logits = new_logits.sum(dim=1)                # (batch_size, self.out_features)

        return final_logits, entropies, sample_entropies
        # return final_logits, current_mixture, entropies
        
    def forward(self, x: torch.Tensor, a: float = 1.0):
        if self.training:
            return self.train_forward(x, a)
        else:
            return self.eval_forward(x)

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(len(x), -1)
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        batch_size = x.shape[0]
        # x has shape (batch_size, in_features)

        current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
        for i in range(self.depth.item()):
            plane_coeffs = self.node_weights.index_select(dim=0, index=current_nodes)       # (batch_size, in_features)
            plane_offsets = self.node_biases.index_select(dim=0, index=current_nodes)       # (batch_size, 1)
            plane_coeff_score = torch.bmm(x.unsqueeze(1), plane_coeffs.unsqueeze(-1))       # (batch_size, 1, 1)

            plane_score = plane_coeff_score.squeeze(-1) + plane_offsets                     # (batch_size, 1)
            plane_choices = (plane_score.squeeze(-1) >= 0).long()                           # (batch_size,)

            platform = torch.tensor(2 ** i - 1, dtype=torch.long, device=x.device)          # (batch_size,)
            next_platform = torch.tensor(2 ** (i+1) - 1, dtype=torch.long, device=x.device) # (batch_size,)
            current_nodes = (current_nodes - platform) * 2 + plane_choices + next_platform  # (batch_size,)

        leaves = current_nodes - next_platform              # (batch_size,)

        out_logits = torch.empty((batch_size, self.out_features), 
                                 dtype=torch.float, device=x.device)
        for l in range(self.n_leaves):
            leaf_indices, = torch.where(leaves == l)
            leaf_logits = torch.matmul(x[leaf_indices], self.w1s[l]) + self.b1s[l].unsqueeze(0)                                               # (1, self.leaf_width)
            activations = self.activation(leaf_logits)           # (1, self.leaf_width)
            leaf_logits = torch.matmul(activations, self.w2s[l]) + self.b2s[l].unsqueeze(0)                                               # (1, self.leaf_width)
            out_logits[leaf_indices] = leaf_logits

        return out_logits, leaves # (..., self.out_features)

    def get_config(self):
        return {
                'depth': self.depth.item(),
                'leaf_width': self.leaf_width,
                }

    # def sigmoid(self, x, a: float):
    #     e = math.exp(1)
    #     return 1 / (1 + e**(-a * x))


def compute_entropy_safe(p: torch.Tensor, minus_p: torch.Tensor) -> torch.Tensor:
    EPSILON = 1e-6
    p = torch.clamp(p, min=EPSILON, max=1-EPSILON)
    minus_p = torch.clamp(minus_p, min=EPSILON, max=1-EPSILON)

    return -p * torch.log(p) - minus_p * torch.log(minus_p)
