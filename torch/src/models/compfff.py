import torch 
import math
from torch import nn
from typing import Optional

class ComplexFFF(nn.Module):
    def __init__(self,
                 in_features: int, leaf_width: int, out_features: int, depth: int):
        super().__init__()
        self.in_features = in_features
        self.leaf_width = leaf_width
        self.out_features = out_features
        self.activation = nn.ReLU()

        if depth < 0 or in_features <= 0 or leaf_width <= 0 or out_features <= 0:
            raise ValueError("input/leaf/output widths and depth must be all positive integers")

        self.n_leaves = 2 ** depth
        self.n_nodes = 2 ** depth - 1
        self.depth = nn.Parameter(torch.tensor(depth, dtype=torch.long), requires_grad=False)

        l1_init_factor = 1.0 / math.sqrt(self.in_features)
        self.node_weights = nn.Parameter(torch.empty((self.n_nodes, 2, in_features), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.node_biases = nn.Parameter(torch.empty((self.n_nodes, 2), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)

        l2_init_factor = 1.0 / math.sqrt(self.leaf_width)
        self.w1s = nn.Parameter(torch.empty((self.n_leaves, in_features, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.b1s = nn.Parameter(torch.empty((self.n_leaves, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w2s = nn.Parameter(torch.empty((self.n_leaves, leaf_width, out_features), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.b2s = nn.Parameter(torch.empty((self.n_leaves, out_features), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)

    def training_forward(self, x: torch.Tensor):
        # x has shape (batch_size, in_features)
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        batch_size = x.shape[0]


        if x.shape[-1] != self.in_features:
            raise ValueError(f"input tensor must have shape (..., {self.in_features})")

        current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
        entropies = torch.zeros((batch_size, self.n_nodes), dtype=torch.float, device=x.device)

        for current_depth in range(self.depth.item()):
            platform = torch.tensor(2 ** current_depth - 1, dtype=torch.long, device=x.device)
            next_platform = torch.tensor(2 ** (current_depth+1) - 1, dtype=torch.long, device=x.device)

            n_nodes = 2 ** current_depth
            current_weights = self.node_weights[platform:next_platform] # (n_nodes, in_features)    
            current_biases = self.node_biases[platform:next_platform]   # (n_nodes, 1)

            boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(0, 1))      # (batch_size, n_nodes)
            boundary_plane_logits = boundary_plane_coeff_scores + current_biases.transpose(0, 1)# (batch_size, n_nodes)
            boundary_effect = torch.sigmoid(boundary_plane_logits)                              # (batch_size, n_nodes)

            not_boundary_effect = 1 - boundary_effect                                   # (batch_size, n_nodes)

            platform_entropies = compute_entropy_safe(
                boundary_effect, not_boundary_effect
            ) # (batch_size, n_nodes)
            entropies[:, platform:next_platform] = platform_entropies   # (batch_size, n_nodes)
                
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
        for i in range(self.n_leaves):
            new_logits[:, i] = torch.matmul(
                element_activations[:, i],
                self.w2s[i]
            ) + self.b2s[i]
        # new_logits has shape (batch_size, self.n_leaves, self.out_features)

        new_logits *= current_mixture.unsqueeze(-1)         # (batch_size, self.n_leaves, self.out_features)
        final_logits = new_logits.sum(dim=1)                # (batch_size, self.out_features)
        
        final_logits = final_logits.view(*original_shape[:-1], self.out_features)   # (..., self.out_features)

        return final_logits, current_mixture, entropies.mean(dim=0)
        
    def forward(self, x: torch.Tensor):
        x = x.view(len(x), -1)
        if self.training:
            return self.training_forward(x)
        else:
            return self.eval_forward(x)

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
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

        new_logits = torch.empty((batch_size, self.out_features), dtype=torch.float, device=x.device)
        for i in range(leaves.shape[0]):
            leaf_index = leaves[i]
            logits = torch.matmul(
                x[i].unsqueeze(0),                  # (1, self.in_features)
                self.w1s[leaf_index]                # (self.in_features, self.leaf_width)
            )                                               # (1, self.leaf_width)
            logits += self.b1s[leaf_index].unsqueeze(-2)    # (1, self.leaf_width)
            activations = self.activation(logits)           # (1, self.leaf_width)
            new_logits[i] = torch.matmul(
                activations,
                self.w2s[leaf_index]
            ).squeeze(-2)                                   # (1, self.out_features)

        return new_logits.view(*original_shape[:-1], self.out_features), leaves # (..., self.out_features)

    def get_config(self):
        return {
                'depth': self.depth.item(),
                'leaf_width': self.leaf_width,
                }

def compute_entropy_safe(p: torch.Tensor, minus_p: torch.Tensor) -> torch.Tensor:
    EPSILON = 1e-6
    p = torch.clamp(p, min=EPSILON, max=1-EPSILON)
    minus_p = torch.clamp(minus_p, min=EPSILON, max=1-EPSILON)

    return -p * torch.log(p) - minus_p * torch.log(minus_p)
