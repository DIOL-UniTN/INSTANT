import os
import torch
import hydra
import dagshub
import mlflow
import logging
import pandas as pd
from pathlib import Path

from utils.fff import CalibTargetTransform
from utils.nn import train_epoch, eval_model
from utils.nn import train_epoch_afff, eval_model_afff
from utils.fff import LeafStats


class ATraining:
    def __init__(self, mlflow_id, entropy_effect:float):
        self.mlflow_id = mlflow_id # 1 - training
        self.exp_name = "ATraining"
        self.entropy_effect = entropy_effect
        self.calib_epochs = 5

    def setup(self, partial_model, optim, partial_loader, device):
        # MLFlow setup
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.log_filename = hydra.core.hydra_config.HydraConfig.get().job.name+'.log'
        self.overrides_config = self.out_dir/'.hydra/overrides.yaml'

        # Model and optim.setup
        self.calib_loader = partial_loader(ltarget_transform=None)
        in_dim = self.calib_loader.in_size[0]*self.calib_loader.in_size[1]
        self.model = partial_model(in_features=self.calib_loader.in_chan*in_dim, 
                           out_features=self.calib_loader.out_dim).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.leaf_criterion = torch.nn.BCEWithLogitsLoss()
        self.optim = optim(self.model.parameters())

    def start_run(self, proj_name, username, mlflow_pass, run_name, debug_level:int):
        # Dagshub and MLFlow setup
        os.environ["MLFLOW_TRACKING_URI"] = f"file:{self.out_dir}/mlruns"
        os.environ["_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT"] = "1001"
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1000"
        # dagshub.init(proj_name, username, mlflow=(debug_level != 3))
        mlflow.environment_variables.MLFLOW_TRACKING_PASSWORD = mlflow_pass

        (mlflow_id, run_name) = (self.mlflow_id, run_name) if debug_level == 0 else (0, 'debug')
        # mlflow.start_run(experiment_id=mlflow_id, run_name=run_name)

        # self.run = mlflow.active_run()
        # logging.info(f"MLFow run ID: {self.run.info.run_id}, status: {self.run.info.status}")

    def end_run(self, metrics, seed:int):
        # Log metrics
        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(self.out_dir/'metrics.csv')

        # Log model
        self.model.to('cpu')
        torch.save(self.model, self.out_dir/'model.pt') # TODO: Add more checkpoints
        torch.save(self.model.state_dict(), self.out_dir/'state_dict.pt') # TODO: Add more checkpoints
        # mlflow.log_artifact(self.out_dir/'model.pt')
        # mlflow.log_artifact(self.out_dir/'state_dict.pt')

        # Log base
        # mlflow.log_param('seed', seed)
        # mlflow.log_artifact(self.out_dir/self.log_filename)
        # mlflow.log_artifact(self.overrides_config)

        # mlflow.end_run()
        # finished_run = mlflow.get_run(self.run.info.run_id)
        # logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")

    def calibrate(self, device, leafstats):
        logging.info(f"Calibrating...")
        for epoch in range(self.calib_epochs):
            train_loss, train_acc = train_epoch(self.model, self.optim, self.calib_loader.train, 
                                                self.criterion, epoch, self.entropy_effect,
                                                device)
            val_loss, val_acc, val_sdecisions, vleaftargets, val_leaf_acc = eval_model(self.model, self.calib_loader.valid, self.criterion, 
                                           device, leafstats) #TODO: Change valid
            test_loss, test_acc, test_sdecisions, _ , test_leaf_acc = eval_model(self.model, self.calib_loader.test, self.criterion, 
                                             device, leafstats) #TODO: Change valid

            # Stats
            logging.info("Val Leaf targets: {}".format(vleaftargets))
            logging.info("Epoch: {} | train acc: {}, train loss: {}, valid acc: {}, valid loss: {}, test acc: {}, test loss: {}".format(
                         epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            logging.info("Val leaf sample: {}, val leaf acc: {}\n test leaf sample: {}, test leaf acc: {}\n".format(
                val_sdecisions.tolist(), val_leaf_acc.tolist(), test_sdecisions.tolist(), test_leaf_acc.tolist()
                ))
            logging.info("Val sample decisions: {}, test sample decisions: {}".format(
                val_sdecisions, test_sdecisions
                ))
            self.model.to(device)
        return vleaftargets

    def run(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.setup(cfg.model, cfg.optim, cfg.loader,
                   cfg.device)
        self.start_run(cfg.proj_name, cfg.username, cfg.mlflow_pass, 
                       cfg.run_name, cfg.debug_level)

        # Param. logging
        # mlflow.log_params({
        #     'depth': self.model.depth.item(),
        #     'leaf_width': self.model.leaf_width,
        #     'task': self.calib_loader.name,
        #     'epochs': cfg.epochs,
        #     })

        # FFF leaf stats
        leafstats = LeafStats(self.model.n_leaves, self.calib_loader.out_dim)

        # Metrics init.
        metrics = {'train_acc': [], 'train_loss': [],
                   'val_acc': [], 'val_loss': [],
                  }
        # Training
    
        # leaftargets = self.calibrate(cfg.device, leafstats)
        # curvies, directs, 
        # leaftargets = [[2, 3, 5], [0, 8], [1, 4, 7], [6, 9]]
        # leaftargets = [[2, 3, 5, 6, 8, 9], [2, 5, 6, 9], [1, 4, 7, 9], [0]]
        # leaftargets = [[0], [1, 5, 8, 3], [2, 6], [4, 7, 9]]
        # leaftargets = [[0,3,5,2,6], [1, 5, 6, 8], [2, 6], [4, 7, 9]]
        # leaftargets = [[0, 3, 5, 6], [1, 2, 5, 7, 8], [2, 6, 0, 4], [4, 7, 9]]
        # leaftargets = [[0,3,5], [6], [1, 8], [2, 5, 7], [3, 8],  [2,6], [4, 7, 9], [9]]
        # (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
        leaftargets = [[0, 2, 6, 8, 9], [1, 3, 4, 5, 7]]
        # leaftargets = [[0, 1, 2], [1, 2], [3], [3, 4, 5]]
        # {0: [0, 3, 5], 1: [1, 8], 6: [2, 6], 7: [4, 7, 9]}
# {6: [0, 4], 2: [1], 1: [2, 5, 7], 3: [3, 8], 0: [6], 9: [9]}


        logging.info("Leaf targets calibrated as: {}". format(leaftargets))
        calib_target_transform = CalibTargetTransform(leaftargets)
        loader  = cfg.loader(ltarget_transform=calib_target_transform)

        for epoch in range(cfg.epochs):
            train_loss, train_acc = train_epoch_afff(self.model, self.optim, loader.train, 
                                                self.criterion, self.leaf_criterion, epoch, self.entropy_effect,
                                                cfg.device)
            val_loss, val_acc, val_sdecisions = eval_model_afff(self.model, loader.valid, self.criterion, 
                                                cfg.device, leafstats) #TODO: Change valid
            test_loss, test_acc, test_sdecisions = eval_model_afff(self.model, loader.test, self.criterion, 
                                                   cfg.device, leafstats) #TODO: Change valid

            logging.info("Epoch: {} | train acc: {}, train loss: {}, valid acc: {}, valid loss: {}, test acc: {}, test loss: {}".format(
                         epoch, train_acc, train_loss, val_acc, val_loss, test_acc, test_loss))
            logging.info("Val sample decisions: {}, test sample decisions: {}".format(
                val_sdecisions, test_sdecisions
                ))
            for log_key in ['train_acc', 'train_loss', 'val_loss', 'val_acc']:
                metrics[log_key].append(eval(log_key))
                # mlflow.log_metric(log_key, eval(log_key), step=epoch)
            self.model.to(cfg.device)

        # # Testing
        test_loss, test_acc = eval_model_afff(self.model, loader.test, self.criterion, 
                                        cfg.device) #TODO: Change valid logging.info("Test acc: {}, Test loss: {}".format(test_acc, test_loss))
        logging.info("TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
        # mlflow.log_metrics({
        #     'test_loss': test_loss,
        #     'test_acc': test_acc,
        #     })

        self.end_run(metrics, cfg.seed)

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
