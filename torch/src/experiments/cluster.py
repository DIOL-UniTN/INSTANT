import os 
import torch
import hydra
import dagshub
import mlflow
import logging
import pandas as pd
from pathlib import Path

from utils.nn import train_epoch, eval_model
from utils.fff import LeafStats


class ClusterTraining:
    def __init__(self, mlflow_id):
        self.mlflow_id = mlflow_id # 1 - training
        self.exp_name = "ClusterTraining"

    def setup(self, partial_model, optim, loader, device):
        # MLFlow setup
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.log_filename = hydra.core.hydra_config.HydraConfig.get().job.name+'.log'
        self.overrides_config = self.out_dir/'.hydra/overrides.yaml'

        # Model and optim.setup
        self.model = partial_model(in_features=loader.in_chan*loader.in_size[0]*loader.in_size[1], 
                           out_features=loader.out_dim).to(device)
        self.criterion = torch.nn.MSELoss()
        self.optim = optim(self.model.parameters())

    def start_run(self, proj_name, username, mlflow_pass, run_name, debug_level:int):
        # Dagshub and MLFlow setup
        os.environ["MLFLOW_TRACKING_URI"] = f"file:{self.out_dir}/mlruns"
        os.environ["_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT"] = "1001"
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1000"
        dagshub.init(proj_name, username, mlflow=(debug_level != 3))
        mlflow.environment_variables.MLFLOW_TRACKING_PASSWORD = mlflow_pass

        (mlflow_id, run_name) = (self.mlflow_id, run_name) if debug_level == 0 else (0, 'debug')
        mlflow.start_run(experiment_id=mlflow_id, run_name=run_name)

        self.run = mlflow.active_run()
        logging.info(f"MLFow run ID: {self.run.info.run_id}, status: {self.run.info.status}")

    def end_run(self, metrics, seed:int):
        logging.info("Success!")
        mlflow.log_param('success', True)

        # Log metrics
        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(self.out_dir/'metrics.csv')

        # Log model
        self.model.to('cpu')
        torch.save(self.model, self.out_dir/'model.pt') # TODO: Add more checkpoints
        torch.save(self.model.state_dict(), self.out_dir/'state_dict.pt') # TODO: Add more checkpoints
        mlflow.log_artifact(self.out_dir/'model.pt')
        mlflow.log_artifact(self.out_dir/'state_dict.pt')

        # Log base
        mlflow.log_param('seed', seed)
        mlflow.log_artifact(self.out_dir/self.log_filename)
        mlflow.log_artifact(self.overrides_config)

        mlflow.end_run()
        finished_run = mlflow.get_run(self.run.info.run_id)
        logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")

    def run(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.setup(cfg.model, cfg.optim, cfg.loader,
                   cfg.device)
        self.start_run(cfg.proj_name, cfg.username, cfg.mlflow_pass, 
                       cfg.run_name, cfg.debug_level)

        # Param. logging
        mlflow.log_params({
            'depth': self.model.depth.item(),
            'leaf_width': self.model.leaf_width,
            'task': cfg.loader.name,
            'epochs': cfg.epochs,
            })

        # FFF leaf stats
        leafstats = LeafStats(self.model.n_leaves, cfg.loader.out_dim)

        # Metrics init.
        metrics = {'train_acc': [], 'train_loss': [],
                   'val_acc': [], 'val_loss': [],
                  }
        # Training
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(100):
            outputs = model(x)
            cluster_probs = F.softmax(outputs, dim=1)  # Convert to cluster probabilities
            
            # Loss: Encourage confident assignments (maximize entropy)
            loss = -torch.mean(torch.sum(cluster_probs * torch.log(cluster_probs + 1e-6), dim=1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')



            # # Testing
            test_loss, test_acc, _, _, test_leaf_acc = eval_model(self.model, cfg.loader.test, self.criterion, 
                                             cfg.device, leafstats) #TODO: Change valid
            logging.info("TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
            mlflow.log_metrics({
                'test_loss': test_loss,
                'test_acc': test_acc,
                })

            self.end_run(metrics, cfg.seed)

            def main(self, cfg):
                try: 
                    self.run(cfg)
                except Exception as e:
                    self.end_failure(e, cfg.seed)

            def end_failure(self, error, seed:int):
                logging.info("Failed!")
                logging.info(e)
                mlflow.log_param('success', False)

                mlflow.log_param('seed', seed)
                mlflow.log_artifact(self.out_dir/self.log_filename)
                mlflow.log_artifact(self.overrides_config)

                mlflow.end_run()
                finished_run = mlflow.get_run(self.run.info.run_id)
                logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")
