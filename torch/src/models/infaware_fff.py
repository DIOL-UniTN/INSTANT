import math
import torch 
from torch import nn
from typing import Optional

from torchvision.transforms import Resize
from utils.fff import ASigmoid, SignActivation

class InfAwareFFF(nn.Module):
    def __init__(self, in_features: int, leaf_width: int, out_features: int, depth: int,
                 sig_alpha: float, inf_sig_alpha: float):
        super().__init__()

        self.in_features = in_features
        self.leaf_width = leaf_width
        self.out_features = out_features
        self.activation = nn.ReLU()
        self.node_activation = ASigmoid.apply
        self.sign = SignActivation.apply

        if depth < 0 or in_features <= 0 or leaf_width <= 0 or out_features <= 0:
            raise ValueError("input/leaf/output widths and depth must be all positive integers")

        self.n_leaves = 2 ** depth
        self.n_nodes = 2 ** depth - 1
        self.depth = depth
        self.sig_alpha = nn.Parameter(torch.tensor(sig_alpha), requires_grad=False)
        self.inf_sig_alpha = nn.Parameter(torch.tensor(inf_sig_alpha), requires_grad=False)

        l1_init_factor = 1.0 / math.sqrt(in_features)
        self.node_weights = nn.Parameter(torch.empty((self.n_nodes, in_features), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.node_biases = nn.Parameter(torch.empty((self.n_nodes, 1), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)

        l2_init_factor = 1.0 / math.sqrt(self.leaf_width)
        self.w1s = nn.Parameter(torch.empty((self.n_leaves, in_features, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.b1s = nn.Parameter(torch.empty((self.n_leaves, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w2s = nn.Parameter(torch.empty((self.n_leaves, leaf_width, out_features), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.b2s = nn.Parameter(torch.empty((self.n_leaves, out_features), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)

    def kill_leaves(self):
        self.w1s.requires_grad, self.w2s.requires_grad = False, False
        self.b1s.requires_grad, self.b2s.requires_grad = False, False

    @torch.no_grad()
    def copy_leaves(self):
        self.w1s[1:], self.w2s[1:] = self.w1s[0], self.w2s[0]
        self.b1s[1:], self.b2s[1:] = self.b1s[0], self.b2s[0]
        
    def forward(self, x: torch.Tensor, warmup: bool = False):
        if self.training and warmup:
            return self.train_warmup_forward(x)
        elif self.training and not warmup:
            return self.train_forward(x)
        else:
            return self.eval_forward(x)

    def train_forward(self, x: torch.Tensor):
        original_shape, batch_size = x.shape, x.size(0)
        x = x.view(len(x), -1)
        x = x.reshape(-1, x.shape[-1])

        current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
        inf_current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
        entropies = torch.zeros((batch_size, self.n_nodes), dtype=torch.float, device=x.device)
        sample_entropies = torch.zeros(self.n_nodes, dtype=torch.float, device=x.device)
        for current_depth in range(self.depth):
            platform = torch.tensor(2 ** current_depth - 1, dtype=torch.long, device=x.device)
            next_platform = torch.tensor(2 ** (current_depth+1) - 1, dtype=torch.long, device=x.device)

            n_nodes = 2 ** current_depth
            current_weights = self.node_weights[platform:next_platform] # (n_nodes, in_features)    
            current_biases = self.node_biases[platform:next_platform]   # (n_nodes, 1)

            boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(0, 1))      # (batch_size, n_nodes)
            boundary_plane_logits = boundary_plane_coeff_scores + current_biases.transpose(0, 1)# (batch_size, n_nodes)

            # Training mixtures
            dir_coeff = torch.sigmoid(boundary_plane_logits)
            other_dir_coeff = 1 - dir_coeff                                   # (batch_size, n_nodes)
            boundary_prob =  dir_coeff.sum(dim=0) / batch_size
            platform_entropies = compute_entropy_safe(dir_coeff, 
                                                      other_dir_coeff)

            entropies[:, platform:next_platform] = platform_entropies   # (batch_size, n_nodes)
            sample_entropies[platform:next_platform] = (-boundary_prob * torch.log2(boundary_prob.sum(dim=0) + 1e-6) +
                                                        -(1 - boundary_prob) * torch.log2(1 - boundary_prob + 1e-6)
                                                        )

            mixture_modifier = torch.cat( # this cat-fu is to interleavingly combine the two tensors
                (other_dir_coeff.unsqueeze(-1), dir_coeff.unsqueeze(-1)),
                dim=-1
            ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)                                               # (batch_size, n_nodes*2, 1)
            current_mixture = current_mixture.view(batch_size, 2 * n_nodes, 
                                                   self.n_leaves // (2 * n_nodes))
            current_mixture.mul_(mixture_modifier)                               
            current_mixture = current_mixture.flatten(start_dim=1, end_dim=2)

            # Inf. aware mixtures
            # inf_dir_coeff = (torch.sign(boundary_plane_logits) + 1) / 2
            inf_dir_coeff = self.sign(boundary_plane_logits)
            inf_other_dir_coeff = 1 - inf_dir_coeff

            inf_mixture_modifier = torch.cat(
                    (inf_other_dir_coeff.unsqueeze(-1), 
                     inf_dir_coeff.unsqueeze(-1)), 
                    dim=-1
                    ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)                                               # (batch_size, n_nodes*2, 1)
            inf_current_mixture = inf_current_mixture.view(batch_size, 2 * n_nodes, 
                                                           self.n_leaves // (2 * n_nodes))
            inf_current_mixture.mul_(inf_mixture_modifier)                               
            inf_current_mixture = inf_current_mixture.flatten(start_dim=1, end_dim=2)

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

        final_logits = (new_logits * current_mixture.unsqueeze(-1)).sum(dim=1)                # (batch_size, self.out_features)
        final_inf_logits = (new_logits * inf_current_mixture.unsqueeze(-1)).sum(dim=1)                # (batch_size, self.out_features)

        return final_logits, final_inf_logits, entropies, sample_entropies, current_mixture

    def train_warmup_forward(self, x: torch.Tensor):
        original_shape, batch_size = x.shape, x.size(0)
        x = x.reshape(-1, x.shape[-1])

        for current_depth in range(self.depth):
            platform = torch.tensor(2 ** current_depth - 1, dtype=torch.long, device=x.device)
            next_platform = torch.tensor(2 ** (current_depth+1) - 1, dtype=torch.long, device=x.device)

            n_nodes = 2 ** current_depth
            current_weights = self.node_weights[platform:next_platform] # (n_nodes, in_features)    
            current_biases = self.node_biases[platform:next_platform]   # (n_nodes, 1)

            boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(0, 1))      # (batch_size, n_nodes)
            boundary_plane_logits = boundary_plane_coeff_scores + current_biases.transpose(0, 1)# (batch_size, n_nodes)

        element_logits = torch.matmul(x, self.w1s.transpose(0, 1).flatten(1, 2))            # (batch_size, self.n_leaves * self.leaf_width)
        element_logits = element_logits.view(batch_size, self.n_leaves, self.leaf_width)    # (batch_size, self.n_leaves, self.leaf_width)
        element_logits += self.b1s.view(1, *self.b1s.shape)                                 # (batch_size, self.n_leaves, self.leaf_width)
        element_activations = self.activation(element_logits)                               # (batch_size, self.n_leaves, self.leaf_width)

        logits = torch.empty((batch_size, self.n_leaves, self.out_features), dtype=torch.float, device=x.device)
        for l in range(self.n_leaves): # TODO: Parallelize
            logits[:, l] = torch.matmul(
                    element_activations[:, l],
                    self.w2s[l]
                    ) + self.b2s[l]
        return logits

    def train_forward_new(self, x: torch.Tensor):
        original_shape, batch_size = x.shape, x.size(0)
        x = x.reshape(-1, x.shape[-1])

        current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
        for current_depth in range(self.depth):
            platform = torch.tensor(2 ** current_depth - 1, dtype=torch.long, device=x.device)
            next_platform = torch.tensor(2 ** (current_depth+1) - 1, dtype=torch.long, device=x.device)

            n_nodes = 2 ** current_depth
            current_weights = self.node_weights[platform:next_platform] # (n_nodes, in_features)    
            current_biases = self.node_biases[platform:next_platform]   # (n_nodes, 1)

            boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(0, 1))      # (batch_size, n_nodes)
            boundary_plane_logits = boundary_plane_coeff_scores + current_biases.transpose(0, 1)# (batch_size, n_nodes)

            # dir_coeff = (torch.sign(boundary_plane_logits) + 1) / 2
            dir_coeff = self.sign(boundary_plane_logits)
            # dir_coeff = torch.sigmoid(boundary_plane_logits)
            other_dir_coeff = 1 - dir_coeff

            mixture_modifier = torch.cat(
                    (other_dir_coeff.unsqueeze(-1), 
                     dir_coeff.unsqueeze(-1)), 
                    dim=-1
                    ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)                                               # (batch_size, n_nodes*2, 1)
            current_mixture = current_mixture.view(batch_size, 2 * n_nodes, 
                                                           self.n_leaves // (2 * n_nodes))
            current_mixture.mul_(mixture_modifier)                               
            current_mixture = current_mixture.flatten(start_dim=1, end_dim=2)

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

        final_logits = (new_logits * current_mixture.unsqueeze(-1)).sum(dim=1)                # (batch_size, self.out_features)
        return final_logits

    def eval_forward(self, x: torch.Tensor):
        original_shape, batch_size = x.shape, x.size(0)
        x = x.view(len(x), -1)
        x = x.reshape(-1, x.shape[-1])

        current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
        out_current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
        for current_depth in range(self.depth):
            platform = torch.tensor(2 ** current_depth - 1, dtype=torch.long, device=x.device)
            next_platform = torch.tensor(2 ** (current_depth+1) - 1, dtype=torch.long, device=x.device)

            n_nodes = 2 ** current_depth
            current_weights = self.node_weights[platform:next_platform] # (n_nodes, in_features)    
            current_biases = self.node_biases[platform:next_platform]   # (n_nodes, 1)

            boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(0, 1))      # (batch_size, n_nodes)
            boundary_plane_logits = boundary_plane_coeff_scores + current_biases.transpose(0, 1)# (batch_size, n_nodes)

            dir_coeff = (torch.sign(boundary_plane_logits) + 1) / 2
            other_dir_coeff = 1 - dir_coeff

            mixture_modifier = (torch.cat((other_dir_coeff.unsqueeze(-1), 
                                           dir_coeff.unsqueeze(-1)), dim=-1)
                                .flatten(start_dim=-2, end_dim=-1).unsqueeze(-1))
            current_mixture = current_mixture.view(batch_size, 2 * n_nodes, 
                                                           self.n_leaves // (2 * n_nodes))
            current_mixture.mul_(mixture_modifier)                               
            current_mixture = current_mixture.flatten(start_dim=1, end_dim=2)

            # FOR OUT
            out_dir_coeff = torch.sigmoid(boundary_plane_logits)
            out_other_dir_coeff = 1 - out_dir_coeff
            out_mixture_modifier = torch.cat( # this cat-fu is to interleavingly combine the two tensors
                (out_other_dir_coeff.unsqueeze(-1), out_dir_coeff.unsqueeze(-1)),
                dim=-1
            ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)                                               # (batch_size, n_nodes*2, 1)
            out_current_mixture = out_current_mixture.view(batch_size, 2 * n_nodes, 
                                                           self.n_leaves // (2 * n_nodes))
            out_current_mixture.mul_(out_mixture_modifier)                               
            out_current_mixture = out_current_mixture.flatten(start_dim=1, end_dim=2)

        element_logits = torch.matmul(x, self.w1s.transpose(0, 1).flatten(1, 2))            # (batch_size, self.n_leaves * self.leaf_width)
        element_logits = element_logits.view(batch_size, self.n_leaves, self.leaf_width)    # (batch_size, self.n_leaves, self.leaf_width)
        element_logits += self.b1s.view(1, *self.b1s.shape)                                 # (batch_size, self.n_leaves, self.leaf_width)
        element_activations = self.activation(element_logits)                               # (batch_size, self.n_leaves, self.leaf_width)

        logits = torch.empty((batch_size, self.n_leaves, self.out_features), dtype=torch.float, device=x.device)
        for l in range(self.n_leaves):
            logits[:, l] = torch.matmul(
                element_activations[:, l],
                self.w2s[l]
                ) + self.b2s[l]

        logits = (logits * current_mixture.unsqueeze(-1)).sum(dim=1)                # (batch_size, self.out_features)
        _, leaves = torch.where(current_mixture == 1)

        return logits, leaves

    def get_config(self):
        return {
                'depth': self.depth,
                'leaf_width': self.leaf_width,
                'train_sig_alpha': self.sig_alpha.item(),
                'inf_sig_alpha': self.inf_sig_alpha.item(),
                }

def compute_entropy_safe(p: torch.Tensor, minus_p: torch.Tensor) -> torch.Tensor:
    EPSILON = 1e-6
    p = torch.clamp(p, min=EPSILON, max=1-EPSILON)
    minus_p = torch.clamp(minus_p, min=EPSILON, max=1-EPSILON)

    return -p * torch.log(p) - minus_p * torch.log(minus_p)
