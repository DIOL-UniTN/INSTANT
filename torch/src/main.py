import os
import torch
import random
import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from dataclasses import dataclass
from hydra.utils import instantiate
from hydra.experimental.callback import Callback
from typing import Any

class MLFlowToCSV(Callback):
    def __init__(self, exp):
        self.exp = exp

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:
        mlflow.set_tracking_uri("file:data/mlruns")
        experiment = mlflow.get_experiment_by_name(self.exp)

        runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        os.makedirs('data/csv_files', exist_ok=True)
        runs_df.to_csv(f"data/csv_files/{self.exp}.csv", index=False)



@dataclass
class Main:
    proj_name: str
    mlflow: object
    mlflow_exp: list
    seed: int
    debug_level: int
    exp: object
    run_name: str
    device: torch.device

    epochs: int
    model: object
    optim: torch.optim.Optimizer
    sched: torch.optim.lr_scheduler.LRScheduler
    loader: object
    sig_sched: object

@hydra.main(config_path="../conf/", config_name="main", version_base='1.2')
def main(cfg: DictConfig):
    # Init RNGs
    torch.set_printoptions(precision=4)
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    cfg = instantiate(cfg)

    # RUN EXPERIMENT
    # set HYDRA DEBUG MODE
    os.environ["HYDRA_FULL_ERROR"] = "1"
    cfg.exp.run(cfg)
    # if cfg.debug_level > 0:
    #     os.environ["HYDRA_FULL_ERROR"] = "1"
    #     cfg.exp.run(cfg)
    # else:
    #     cfg.exp.main(cfg)

if __name__ == "__main__":
    main()
