import os
import hydra
import yaml
from pathlib import Path
import torch

class  ModelToC:
    def __init__(self, model_file: str, in_file: str, out_file: str):
        self.model_file = model_file
        self.in_file = in_file
        self.out_file = out_file

    def setup(self, cfg):
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.bins_dir = self.out_dir/"param_bins"
        self.bins_dir.mkdir(parents=True, exist_ok=True)
        self.model = torch.load(self.model_file).cpu()
        self.input = torch.load(self.in_file).cpu()
        self.output = torch.load(self.out_file).cpu()

    def tensor_to_c_array(self, x: torch.Tensor, name: str):
        x = str(x.tolist()).replace("[", "{").replace("]", "}").replace("},", "},\n")
        return f"{name} = {x};\n"

    @torch.no_grad()
    def write_model(self):
        # Model Config & Params
        with open(self.out_dir/"model.h", "w") as f:
            # Includes
            f.write(f'#include "mem.h"\n\n')
            # Config
            for key, val in self.model.get_config().items():
                f.write(f"#define {key.upper()} {val}\n")
            # Params
            fc1_w, fc1_b = self.model.fc1.weight, self.model.fc1.bias
            fc2_w, fc2_b = self.model.fc2.weight, self.model.fc2.bias
            fc1_w = self.tensor_to_c_array(fc1_w, "\n__big float fc1_w[WIDTH][IN_FEATURES]")
            fc1_b = self.tensor_to_c_array(fc1_b, "\n__big float fc1_b[WIDTH]")
            fc2_w = self.tensor_to_c_array(fc2_w, "\n__big float fc2_w[OUT_FEATURES][WIDTH]")
            fc2_b = self.tensor_to_c_array(fc2_b, "\n__big float fc2_b[OUT_FEATURES]")
            # DATA
            input = self.tensor_to_c_array(self.input, "\n__big float input[IN_FEATURES]")
            hidden = "\n__big float hidden[WIDTH];"
            output = "\n\n__big float output[OUT_FEATURES];"
            py_output = self.tensor_to_c_array(self.output, "\n\n__big float py_output[OUT_FEATURES]")
            for p in [fc1_w, fc1_b, fc2_w, fc2_b, input, hidden, output, py_output]:
                f.write(p)

    @torch.no_grad()
    def write_model_small(self, w: int = 16):
        # Model Config & Params
        with open(self.out_dir/"model.h", "w") as f:
            # Includes
            f.write(f'#include "mem.h"\n\n')
            # Config
            for key, val in self.model.get_config().items():
                val = w if key == "width" else val
                f.write(f"#define {key.upper()} {val}\n")
            # Params
            fc1_w, fc1_b = self.model.fc1.weight[:w], self.model.fc1.bias[:w]
            fc2_w, fc2_b = self.model.fc2.weight[:, :w], self.model.fc2.bias[:]
            fc1_w = self.tensor_to_c_array(fc1_w, "\n__big float fc1_w[WIDTH][IN_FEATURES]")
            fc1_b = self.tensor_to_c_array(fc1_b, "\n__big float fc1_b[WIDTH]")
            fc2_w = self.tensor_to_c_array(fc2_w, "\n__big float fc2_w[OUT_FEATURES][WIDTH]")
            fc2_b = self.tensor_to_c_array(fc2_b, "\n__big float fc2_b[OUT_FEATURES]")
            # DATA
            input = self.tensor_to_c_array(self.input, "\n__big float input[IN_FEATURES]")
            output = self.tensor_to_c_array(self.output, "\n__big float py_output[OUT_FEATURES]")
            for p in [fc1_w, fc1_b, fc2_w, fc2_b, input, output]:
                f.write(p)

    def run(self, cfg):
        self.setup(cfg)

        self.write_model()

        self.model.to('cpu')
        torch.save(self.model, os.path.join(self.out_dir, f'model.pt')) # TODO: Add more checkpoints
        torch.save(self.model.state_dict(), os.path.join(self.out_dir, f'state_dict.pt')) # TODO: Add more checkpoints
