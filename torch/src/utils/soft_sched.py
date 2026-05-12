import torch

class SoftSchedLinear: # TODO: Cosine annealing
    def __init__(self, temp: torch.nn.Parameter, epochs: int):
        self.temp = temp
        self.epochs = epochs
        self.base_temp = temp.item()
        self.device = self.temp.device
        self.epoch = 0

    def step(self):
        if self.epoch % 10 == 0:
            cur_temp = self.temp.item()
            self.temp.data.copy_(torch.tensor(cur_temp/2, device=self.device).clamp(max=100.0))
        self.epoch += 1

    def get_base_config(self):
        return {}

    def get_config(self):
        return self.get_base_config()
