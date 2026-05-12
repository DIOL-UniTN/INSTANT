import torch
import math
from torch import nn
# 2**2, 2**1, 2**0
class FeatFFF(nn.Module):
    def __init__(self, in_features: int, leaf_width: int, out_features: int, depth: int, leaf_dropout:int, fff_reg_effect, node_act: str, complex_start:bool):
        super().__init__()
        self.model_name = "FeatFFF"
        self.in_features = in_features
        self.leaf_width = leaf_width
        self.out_features = out_features
        self.activation = nn.ReLU()
        self.leaf_dropout = leaf_dropout
        self.complex_start = complex_start
        self.node_width = 0.1 if complex_start else 0.2
        self.n_act = node_act

        if depth < 0 or in_features <= 0 or leaf_width <= 0 or out_features <= 0:
            raise ValueError("input/leaf/output and depth must be all positive integers.")
        if leaf_dropout < 0:
            raise ValueError("Leaf dropout cannot be negative.")
        if fff_reg_effect == 0.0 and leaf_dropout != 0.0:
            raise ValueError("Leaf dropout and diff. FFF regularization does not work together.")

        self.fff_reg_effect = fff_reg_effect
        fff_reg_param, reg_grad = (fff_reg_effect, False) if fff_reg_effect else (1.0, True)
        self.fff_reg_param = fff_reg_effect if fff_reg_effect else nn.Parameter(torch.tensor(fff_reg_effect), requires_grad=True)

        self.depth = nn.Parameter(torch.tensor(depth, dtype=torch.long), requires_grad=False)
        self.n_leaves = 2 ** depth
        self.n_nodes = 2 ** depth - 1

        l1_init_factor = 1.0 / math.sqrt(self.in_features)
        self.node_weights, self.node_biases = nn.ParameterList(), nn.ParameterList()
        feat_start = 256
        for d in range(depth):
            cur_n_nodes = 2**d
            cur_node_width = 2**(depth-d) if complex_start else 2**(d)
            cur_node_weights = nn.Parameter(torch.empty((cur_node_width, cur_n_nodes, in_features), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
            cur_node_biases = nn.Parameter(torch.empty((cur_node_width, cur_n_nodes, 1), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
            self.node_weights.append(cur_node_weights)
            self.node_biases.append(cur_node_biases)

        l2_init_factor = 1.0 / math.sqrt(self.leaf_width)
        self.w1s = nn.Parameter(torch.empty((self.n_leaves, in_features, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.b1s = nn.Parameter(torch.empty((self.n_leaves, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w2s = nn.Parameter(torch.empty((self.n_leaves, leaf_width, out_features), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.b2s = nn.Parameter(torch.empty((self.n_leaves, out_features), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)

    def training_forward(self, x: torch.Tensor, itr_effect:float=1.0, ldropout_epoch:float=0.0):
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
            current_weights = self.node_weights[current_depth] # (n_nodes, in_features)    
            current_biases = self.node_biases[current_depth]   # (n_nodes, 1)

            boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(1, 2))
            boundary_plane_logits = (boundary_plane_coeff_scores + current_biases.transpose(1, 2)).mean(dim=0) # (batch_size, n_nodes)
            boundary_effect = self.node_act(boundary_plane_logits)                              # (batch_size, n_nodes)

            not_boundary_effect = 1 - boundary_effect                                   # (batch_size, n_nodes)

            platform_entropies = compute_entropy_safe(
                boundary_effect, not_boundary_effect
            ) # (batch_size, n_nodes)
            entropies[:, platform:next_platform] = platform_entropies	# (batch_size, n_nodes)

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

        # Another one
        current_mixture **= self.fff_reg_param*itr_effect

        # And another one
        if self.leaf_dropout:
            max_n_indices = int(current_mixture.size(1)*self.leaf_dropout)
            n_indices = torch.randint(0, max_n_indices+1, (1, ))
            mask = torch.argsort(torch.rand(current_mixture.shape), dim=1)[:, :n_indices]
            current_mixture[torch.arange(current_mixture.size(0)).unsqueeze(1), mask] = 0.0

        new_logits *= current_mixture.unsqueeze(-1)         # (batch_size, self.n_leaves, self.out_features)
        final_logits = new_logits.sum(dim=1)                # (batch_size, self.out_features)
        final_logits = final_logits.view(*original_shape[:-1], self.out_features)   # (..., self.out_features)

        return final_logits, entropies.mean(dim=0), current_mixture, new_logits
        
    def forward(self, x: torch.Tensor, itr_effect:float=1.0, ldropout_epoch:float=0.0):
        x = x.view(len(x), -1)
        if self.training:
            return self.training_forward(x, itr_effect, ldropout_epoch)
        return self.eval_forward(x)

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        batch_size = x.shape[0]

        current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
        for i in range(self.depth.item()):
            plane_coeffs = self.node_weights[i].index_select(dim=1, index=current_nodes)       # (batch_size, in_features)
            plane_offsets = self.node_biases[i].index_select(dim=1, index=current_nodes)       # (batch_size, 1)
            plane_coeff_score = torch.bmm(x.unsqueeze(1), plane_coeffs.permute(1, 2, 0))       # (batch_size, 1, 1)
            plane_score = (plane_coeff_score.permute(2, 0, 1) + plane_offsets).mean(dim=0)                     # (batch_size, 1)
            plane_choices = (plane_score.squeeze(-1) >= 0).long()                           # (batch_size,)

            current_nodes = current_nodes * 2 + plane_choices # (batch_size,)

        leaves = current_nodes
        new_logits = torch.empty((batch_size, self.out_features), dtype=torch.float, device=x.device)
        for i in range(leaves.shape[0]):
            leaf_index = leaves[i]
            logits = torch.matmul(
                x[i].unsqueeze(0),                  # (1, self.in_features)
                self.w1s[leaf_index]                # (self.in_features, self.leaf_width)
            )                                               # (1, self.leaf_width)
            logits += self.b1s[leaf_index].unsqueeze(-2)    # (1, self.leaf_width)
            activations = self.activation(logits)           # (1, self.leaf_width)
            new_logits[i] = (torch.matmul(
                activations,
                self.w2s[leaf_index]
            ) + self.b2s[leaf_index]).squeeze(-2)                                   # (1, self.out_features)

        return new_logits.view(*original_shape[:-1], self.out_features), leaves

    def node_act(self, x):
        if self.n_act == 'sigmoid':
            return torch.sigmoid(x)
        elif self.n_act == 'hardsigmoid':
            return torch.nn.functional.sigmoid(x)
        elif self.n_act == 'tanh':
            return (1 + torch.tanh(x))/2

def compute_entropy_safe(p: torch.Tensor, minus_p: torch.Tensor) -> torch.Tensor:
	EPSILON = 1e-6
	p = torch.clamp(p, min=EPSILON, max=1-EPSILON)
	minus_p = torch.clamp(minus_p, min=EPSILON, max=1-EPSILON)

	return -p * torch.log(p) - minus_p * torch.log(minus_p)
