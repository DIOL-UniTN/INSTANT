import math
import torch

TINFO = torch.finfo(torch.float)

MNIST_LEAF_TRANSFROM = {
        0: 0, 6: 0, 8: 0, 9: 0, 1: 1, 2: 2, 3: 2, 5: 2, 4: 3,7: 3,
        }

def leaf_target_transform(target):
    return MNIST_LEAF_TRANSFROM[target]

class CalibTargetTransform:
    def __init__(self, leaftargets):
        self.leaftargets = leaftargets

    def __call__(self, target):
        x = [float(target in leaftarget) for leaftarget in self.leaftargets]
        return x

class LeafStats:
    def __init__(self, n_leaves, n_classes):
        self.n_leaves = n_leaves
        self.n_classes = n_classes

    def sample(self, leaves):
        return torch.tensor([(leaves==i).sum().item() for i in range(self.n_leaves)])

    def correct(self, leaves, correct):
        return torch.tensor([correct[leaves==i].sum().item() for i in range(self.n_leaves)])

    def calib(self, targets, leaves):
        all_leaf_classes = [targets[leaves == i] for i in range(self.n_leaves)]
        leaf_classes = []
        for leaf_class in all_leaf_classes:
            cur_leaf_classes = [(leaf_class == i).sum().item() 
                                for i in range(self.n_classes)]
            leaf_classes.append(torch.tensor(cur_leaf_classes))
        leaf_classes = [leaf_class/leaf_class.sum() for leaf_class in leaf_classes]
        leaf_classes = [torch.where(leaf_class > 0.1)[0].tolist() for leaf_class in leaf_classes]
        return leaf_classes

class ASigmoidSchedLinear: # TODO: Cosine annealing
    def __init__(self, alpha: torch.nn.Parameter, temp: float, warmup_rate: float, 
                 warmup: object, epochs: int):
        self.alpha = alpha
        self.temp = temp
        self.epochs = epochs
        self.warmup = warmup
        self.warmup_rate = warmup_rate
        self.warmup_epochs = warmup_rate * epochs
        self.device = self.alpha.device
        self.epoch = 0

    def step(self):
        alpha = self.warmup()
        if self.epoch >= self.warmup_epochs:
            alpha = 100.0 * (self.epoch/self.epochs) * self.temp
        self.alpha.data.copy_(torch.tensor(alpha, device=self.device).clamp(max=100.0))
        self.epoch += 1

    def get_base_config(self):
        return {
                'asig_temp': self.temp,
                'asig_warmupr': self.warmup_rate,
                }

    def get_config(self):
        return self.get_base_config()

class ASigmoidSchedEaseIn(ASigmoidSchedLinear): # TODO: Cosine annealing
    def __init__(self, alpha: torch.nn.Parameter, temp: float, warmup_rate: float, 
                 warmup: float, epochs: int):
        super().__init__(alpha, temp, warmup_rate, warmup, epochs)

    def step(self):
        alpha = 1.0
        if self.epoch >= self.warmup_epochs:
            alpha = 100.0 * ((self.epoch/self.epochs) ** self.temp)
        self.alpha.data.copy_(torch.tensor(alpha, device=self.device))
        self.epoch += 1

class ASigmoidSchedEaseOut(ASigmoidSchedLinear): # TODO: Cosine annealing
    def __init__(self, alpha: torch.nn.Parameter, temp: float, warmup_rate: float, 
                 warmup: float, epochs: int):
        super().__init__(alpha, temp, warmup_rate, warmup, epochs)

    def step(self):
        alpha = self.warmup()
        if self.epoch >= self.warmup_epochs:
            alpha = 100.0 * (1 - (1 - self.epoch/self.epochs) ** self.temp)
        self.alpha.data.copy_(torch.tensor(alpha, device=self.device))
        self.epoch += 1

class ASigmoidSchedEaseOutRestart(ASigmoidSchedLinear): # TODO: Cosine annealing
    def __init__(self, alpha: torch.nn.Parameter, temp: float, warmup_rate: float, 
                 warmup: float, restart_epoch: int, epochs: int):
        super().__init__(alpha, temp, warmup_rate, warmup, epochs)
        self.restart_epoch = restart_epoch
        self.base_alpha = self.alpha.item()

    def step(self):
        alpha = self.base_alpha + 100.0 * (1 - (1 - self.epoch/self.restart_epoch) ** self.temp)
        self.alpha.data.copy_(torch.tensor(alpha, device=self.device))
        self.epoch += 1
        if (self.epoch % (self.restart_epoch+1)) == 0:
            self.epoch = 0

    def get_config(self):
        conf = {
                "sig_sched_name": "ease out restarts",
                "restart_epoch": self.restart_epoch
                }
        return conf | self.get_base_config()

class ASigmoidSchedEaseInRestart(ASigmoidSchedLinear): # TODO: Cosine annealing
    def __init__(self, alpha: torch.nn.Parameter, temp: float, warmup_rate: float, 
                 warmup: float, restart_epoch: int, epochs: int):
        super().__init__(alpha, temp, warmup_rate, warmup, epochs)
        self.restart_epoch = restart_epoch
        self.base_alpha = self.alpha.item()

    def step(self):
        alpha = self.base_alpha + 100.0 * ((self.epoch/self.restart_epoch) ** self.temp)
        self.alpha.data.copy_(torch.tensor(alpha, device=self.device))
        self.epoch += 1
        if (self.epoch % (self.restart_epoch+1)) == 0:
            self.epoch = 0

    def get_config(self):
        conf = {
                "sig_sched_name": "ease-in-restarts",
                "restart_epoch": self.restart_epoch
                }
        return conf | self.get_base_config()

class ASigmoidSchedCosineAnnealing(ASigmoidSchedLinear): # TODO: Cosine annealing
    def __init__(self, alpha: torch.nn.Parameter, T_max: int, eta_min: float,
                 warmup_rate: float, warmup: float, epochs: int):
        super().__init__(alpha, 0, warmup_rate, warmup, epochs)
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_alpha = self.alpha.item()

    def step(self): # Torch implementation
        if self.epoch == 10:
            breakpoint()
        if self.epoch == 0:
            alpha = 1.0
        elif self.epoch == 1: 
            alpha = (
                    self.eta_min
                    + (self.base_alpha - self.eta_min)
                    * (1 + math.cos((self.epoch) * math.pi / self.T_max))
                    / 2
                    )
        elif (self.epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            alpha = (
                    self.alpha
                    + (self.base_alpha - self.eta_min) * \
                            (1 - math.cos(math.pi / self.T_max)) / 2
                    )
        else:
            alpha = (
                    (1 + math.cos(math.pi * self.epoch / self.T_max))
                    / (1 + math.cos(math.pi * (self.epoch - 1) / self.T_max))
                    * (self.alpha - self.eta_min)
                    + self.eta_min
                    )
        self.epoch += 1
        self.alpha.data.copy_(torch.tensor(1 / alpha, device=self.device))

    def get_config(self):
        conf = {
                "T_max": self.T_max,
                "eta_min": self.eta_min,
                "base_alpha": self.base_alpha,
                }
        return conf | self.get_base_config()
    # def step(self):
    #     alpha = self.warmup()
    #     if self.epoch >= self.warmup_epochs:
    #         self.cur_alpha = self.alpha + (1 + torch.cos((self.epoch/self.tmax) * torch.pi))
    #     self.alpha.data.copy_(torch.tensor(self.cur_alpha, device=self.device))
    #     self.epoch += 1

class WarmUp:
    def __init__(self, epochs: int):
        self.epochs = epochs
        self.epoch = 0
    
    def __call__(self):
        return 1.0

class WarmUpLinear(WarmUp):
    def __init__(self, epochs: int):
        super().__init__(epochs)
    
    def __call__(self):
        alpha =  self.epoch/self.epochs
        self.epochs += 1
        return alpha

class WarmUpEaseOut(WarmUp):
    def __init__(self, epochs: int):
        super().__init__(epochs)
    
    def __call__(self):
        alpha = (self.epoch/self.epochs - 1) ** 2
        self.epoch += 1
        return alpha

class WarmUpEaseIn(WarmUp):
    def __init__(self, epochs):
        super().__init__(epochs)
    
    def __call__(self):
        alpha = (self.epoch/self.epochs) ** 2
        self.epoch += 1
        return alpha

    def get_config(self):
        return {
                "warmup_rate": rate
                }

class ASigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        result = 1 / (1 + torch.exp(-a * x))  # Steep sigmoid
        ctx.save_for_backward(x, a)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_tensors
        term1, term2 = (a * torch.exp(-a * x)), (1 + torch.exp(-a * x)**2)
        term = (term1.clamp(min=TINFO.min, max=TINFO.max) /
                term2.clamp(min=TINFO.min, max=TINFO.max))
        grad_input = grad_output * term
        grad_a = None
        return grad_input, grad_a
    
class SignActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (torch.sign(x) + 1) / 2  # Forward pass

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # return grad_output 
        return grad_output / x.abs()

    # @staticmethod
    # def backward(ctx, grad_output):
    #     x, = ctx.saved_tensors
    #     grad_input = grad_output * (torch.exp(-x) / (1 + torch.exp(-x)**2))
    #     return grad_input

