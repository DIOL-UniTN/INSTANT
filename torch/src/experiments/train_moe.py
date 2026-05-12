import os
import hydra
import dagshub
import mlflow
import logging
from pathlib import Path
import pandas as pd

import torch
from utils.nn_moe import train_epoch_moe, eval_model_moe
from utils.hydra import get_multirun_swept_overrides

class TrainMoE:
    def __init__(self, mlflow_id):
        self.mlflow_id = mlflow_id
        self.exp_name = "TrainingMoE"

    def get_config(self):
        exp_config = {
                      'exp_name': self.exp_name,
                     }
        return exp_config | self.model.get_config() | self.sig_sched.get_config()

    def setup(self, partial_model, optim, sig_sched, loader, device):
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.log_filename = hydra.core.hydra_config.HydraConfig.get().job.name+'.log'
        self.swept_overrides = get_multirun_swept_overrides(self.out_dir)
        self.overrides_config = self.out_dir/'.hydra/overrides.yaml'

        # Model and optim.setup
        self.model = partial_model(in_features=loader.in_chan*loader.in_size[0]*loader.in_size[1], 
                           out_features=loader.out_dim).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = optim(self.model.parameters())
        self.sig_sched = sig_sched(self.model.gate.temp)

    def get_mlflow_run_name(self, cfg):
        # Run name considering sweep
        run_name = ""
        for override in self.swept_overrides:
            try:
                override_val = eval(f"cfg.{override}")
            except:
                override_val = eval(f"self.{override}")
            override_name = override.split('.')[-1]
            run_name += f"{override_name}={override_val}"
        return run_name if len(run_name) else None

    def start_run(self, proj_name, username, mlflow_pass, run_name, debug_level:int):
        # Dagshub and MLFlow setup
        os.environ["MLFLOW_TRACKING_URI"] = f"file:{self.out_dir}/mlruns"
        os.environ["_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT"] = "1001"
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1000"
        dagshub.init(proj_name, username, mlflow=(debug_level != 2))
        mlflow.environment_variables.MLFLOW_TRACKING_PASSWORD = mlflow_pass
        mlflow.start_run(experiment_id=(self.mlflow_id if debug_level == 0 else 0), run_name=run_name)
        self.run = mlflow.active_run()
        logging.info(f"MLFow run ID: {self.run.info.run_id}, status: {self.run.info.status}")

    def end_run(self, train_metrics, val_metrics, seed:int):
        # Log metrics
        df = pd.DataFrame.from_dict(train_metrics)
        df.to_csv(self.out_dir/'train_metrics.csv')
        df = pd.DataFrame.from_dict(val_metrics)
        df.to_csv(self.out_dir/'val_metrics.csv')

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
        self.setup(cfg.model, cfg.optim, cfg.sig_sched, cfg.loader, cfg.device)
        self.start_run(cfg.proj_name, cfg.username, cfg.mlflow_pass, 
                       self.get_mlflow_run_name(cfg), cfg.debug_level)

        # Param. logging
        mlflow.log_params(self.get_config())

        # Metrics init.
        train_metrics = {'train_acc': [], 'train_loss': [],
                         }
        val_metrics = {'val_loss': [], 'val_acc': [], 'gate_temp': []
                      }
        # Training
        for epoch in range(cfg.epochs):
            train_loss, train_acc = train_epoch_moe(self.model, self.optim, cfg.loader.train, 
                                                   self.criterion, epoch, cfg.device)
            # Logging
            logging.info("Epoch: {} | TRAIN | acc: {:.4f}, loss: {:.4f}".format(
                         epoch, train_acc, train_loss))
            val_loss, val_acc = eval_model_moe(self.model, cfg.loader.valid, self.criterion, cfg.device) #TODO: Change valid
            gate_temp = self.model.gate.temp.item()
            logging.info("       {} | VALID | acc: {:.4f}, loss: {:.4f}, gate temp: {:.4f}".format(
                         ' '*len(str(epoch)), val_acc, val_loss, gate_temp))
            for log_key in train_metrics:
                train_metrics[log_key].append(eval(log_key))
                mlflow.log_metric(log_key, eval(log_key), step=epoch)
            for log_key in val_metrics:
                val_metrics[log_key].append(eval(log_key))
                mlflow.log_metric(log_key, eval(log_key), step=epoch)
            if epoch % 5 == 0 or epoch == cfg.epochs - 1:
                test_loss, test_acc = eval_model_moe(self.model, cfg.loader.test, self.criterion, cfg.device) #TODO: Change valid
                logging.info("       {} | TEST | acc: {:.4f}, loss: {:.4f}".format(
                             ' '*len(str(epoch)), test_acc, test_loss))
            self.sig_sched.step()

        # Testing
        test_loss, test_acc = eval_model_moe(self.model, cfg.loader.test, self.criterion, cfg.device) #TODO: Change valid logging.info("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
        logging.info("TEST | acc: {:.4f}, loss: {:.4f}".format(
                     test_acc, test_loss))
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_acc': test_acc,
            })


        self.end_run(train_metrics, val_metrics, cfg.seed)

    def main(self, cfg):
        try: 
            self.run(cfg)
        except Exception as e:
            self.end_failure(e, cfg.seed)

    def end_failure(self, error, seed:int):
            logging.info("Failed!")
            logging.info(error)
            mlflow.log_param('success', False)

            mlflow.log_param('seed', seed)
            mlflow.log_artifact(self.out_dir/self.log_filename)
            mlflow.log_artifact(self.overrides_config)

            mlflow.end_run()
            finished_run = mlflow.get_run(self.run.info.run_id)
            logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")
