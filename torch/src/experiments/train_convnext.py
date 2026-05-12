import os
import hydra
import dagshub
import mlflow
import logging
from pathlib import Path
import pandas as pd

import torch
from models.convnext import convnext_tiny
from utils.nn_ff import train_epoch_ff, eval_model_ff
from utils.hydra import get_multirun_swept_overrides

class TrainConvNeXt:
    def __init__(self, mlflow_id):
        self.mlflow_id = mlflow_id
        self.exp_name = "TrainConvNeXt"

    def setup(self, optim, loader, device):
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) # pyright: ignore
        self.log_filename = hydra.core.hydra_config.HydraConfig.get().job.name+'.log' # pyright: ignore
        self.swept_overrides = get_multirun_swept_overrides(self.out_dir)
        self.overrides_config = self.out_dir/'.hydra/overrides.yaml'

        # Model and optim.setup
        self.model = convnext_tiny(pretrained=False, in_22k=False, 
                                   in_chans=loader.in_chan,
                                   num_classes=loader.out_dim).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optim = optim(self.model.parameters())

    def start_run(self, proj_name, username, token, run_name, debug_level:int, 
                  server:str, exp: str):
        # Dagshub and MLFlow setup
        os.environ["MLFLOW_TRACKING_URI"] = f"file:data/mlruns"
        os.environ["_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT"] = "1001"
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1000"
        if server == "dagshub":
            mlflow.environment_variables.MLFLOW_TRACKING_PASSWORD = token # pyright: ignore
            dagshub.init(proj_name, username, 
                         mlflow=(debug_level < 2) and (server=="dagshub"))

        if exp:
            mlflow.set_experiment(exp)
        run_name = run_name if debug_level == 0 else 'debug'
        mlflow.start_run(run_name=run_name)
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
        mlflow.log_artifact(str(self.out_dir/'model.pt'))
        mlflow.log_artifact(str(self.out_dir/'state_dict.pt'))

        # Log base
        mlflow.log_param('seed', seed)
        mlflow.log_artifact(self.out_dir/self.log_filename)
        mlflow.log_artifact(str(self.overrides_config))

        mlflow.end_run()
        finished_run = mlflow.get_run(self.run.info.run_id)
        logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")

    def run(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.setup(cfg.optim, cfg.loader,
                   cfg.device)
        self.start_run(cfg.proj_name, cfg.mlflow.username, cfg.mlflow.token, cfg.run_name, 
                       cfg.debug_level, cfg.mlflow.server, cfg.mlflow.exp)

        # Param. logging
        mlflow.log_params({
            'task': cfg.loader.name,
            'epochs': cfg.epochs,
        })

        # Metrics init.
        train_metrics = {'train_acc': [], 'train_loss': [],
                         }
        val_metrics = {'val_loss': [], 'val_acc': [], 
                       }
        # Training
        for epoch in range(cfg.epochs):
            train_loss, train_acc = train_epoch_ff(self.model, self.optim, cfg.loader.train, 
                                                   self.criterion, epoch, cfg.device)
            # Logging
            logging.info("Epoch: {} | TRAIN | acc: {:.4f}, loss: {:.4f}".format(
                epoch, train_acc, train_loss))
            if epoch % 5 == 0 or epoch == cfg.epochs - 1:
                # val_loss, val_acc = eval_model_ff(self.model, cfg.loader.valid, self.criterion, cfg.device) #TODO: Change valid
                test_loss, test_acc = eval_model_ff(self.model, cfg.loader.test, self.criterion, cfg.device) #TODO: Change valid
                # logging.info("       {} | VALID | acc: {:.4f}, loss: {:.4f}".format(
                    #              ' '*len(str(epoch)), val_acc, val_loss))
                logging.info("       {} | TEST | acc: {:.4f}, loss: {:.4f}".format(
                    ' '*len(str(epoch)), test_acc, test_loss))
                # for log_key in val_metrics:
                    #     val_metrics[log_key].append(eval(log_key))
                #     mlflow.log_metric(log_key, eval(log_key), step=epoch)
            for log_key in train_metrics:
                train_metrics[log_key].append(eval(log_key))
                mlflow.log_metric(log_key, eval(log_key), step=epoch)

        # Testing
        test_loss, test_acc = eval_model_ff(self.model, cfg.loader.test, self.criterion, cfg.device) #TODO: Change valid logging.info("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
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
        mlflow.log_artifact(str(self.overrides_config))

        mlflow.end_run()
        finished_run = mlflow.get_run(self.run.info.run_id)
        logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")
