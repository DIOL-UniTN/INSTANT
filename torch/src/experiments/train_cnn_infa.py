import os 
import torch
import hydra
import dagshub
import mlflow
import logging
import pandas as pd
from pathlib import Path

from utils.nn import train_epoch_infa, train_epoch_warmup, eval_model_cnn
from utils.fff import LeafStats

class TrainingCNNInfAware:
    def __init__(self, mlflow_id, sched):
        self.mlflow_id = mlflow_id # 1 - training
        self.exp_name = "TrainingCNNInfAdaptive"
        self.sched = sched

    def get_config(self):
        exp_config = {
                      'exp_name': self.exp_name,
                      'lrsched_step_size': self.sched.step_size,
                      'lrsched_gamma': self.sched.gamma,
                      'init_lr': self.optim.defaults['lr'],
                     }
        return exp_config | self.model.get_config() | self.sig_sched.get_config()

    def setup(self, partial_model, optim, sched, sig_sched, loader, device):
        # MLFlow setup
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.log_filename = hydra.core.hydra_config.HydraConfig.get().job.name+'.log'
        self.overrides_config = self.out_dir/'.hydra/overrides.yaml'

        # Model and optim.setup
        self.model = partial_model(in_channels=loader.in_chan, 
                                   in_features=loader.in_size[0]*loader.in_size[1], 
                                   out_features=loader.out_dim).to(device)
        self.optim = optim(self.model.parameters())
        self.sched = sched(self.optim)
        self.sig_sched = sig_sched(self.model.classifier.sig_alpha)
        self.loader = loader
        self.device = device


        # Model metrics init.
        self.metrics = {
            'train_acc': [], 'train_loss': [], 'train_inf_acc': [], 
            'train_inf_loss': [], 'val_acc': [], 'val_loss': [],
            'train_sample_entropy': [], 'sigmoid_alpha': [],
            'train_entropy': [], 'lr': [],
        }

    def start_run(self, proj_name, username, token, run_name, debug_level:int, 
                  server:str, exp: str):
        # Dagshub and MLFlow setup
        os.environ["MLFLOW_TRACKING_URI"] = f"file:data/mlruns"
        os.environ["_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT"] = "1001"
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1000"
        if server == "dagshub":
            mlflow.environment_variables.MLFLOW_TRACKING_PASSWORD = token
            dagshub.init(proj_name, username, 
                         mlflow=(debug_level < 2) and (server=="dagshub"))

        expid = self.mlflow_id
        if exp:
            mlflow.set_experiment(exp)
        run_name = run_name if debug_level == 0 else 'debug'
        mlflow.start_run(run_name=run_name)
        self.run = mlflow.active_run()
        logging.info(f"MLFow run ID: {self.run.info.run_id}, status: {self.run.info.status}")

    def end_run(self, seed:int):
        logging.info("Success!")
        mlflow.log_param('success', True)

        # Log metrics
        df = pd.DataFrame.from_dict(self.metrics)
        df.to_csv(self.out_dir/'metrics.csv')

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
        finished_run = mlflow.get_run(str(self.run.info.run_id))
        logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")

    def log_epoch(self, train, train_inf, val, epoch, best_val, testloader, val_preleaves, device):
        train_loss, train_acc, train_entropy, train_sample_entropy = train
        train_inf_loss, train_inf_acc, tsdecisions, tleaftargets,  tleaf_acc, _, _ = train_inf 
        val_loss, val_acc, vsdecisions, vleaftargets, vleaf_acc, val_leaves, val_corrects = val 
        sigmoid_alpha, (lr,) = self.sig_sched.alpha.item(), self.sched.get_lr()

        changed_indices = val_preleaves != val_leaves
        change_mag = (val_preleaves[changed_indices] - val_leaves[changed_indices]).abs() // 2
        change_counts = [(change_mag == i).sum().item() for i in range(self.model.classifier.depth)]
        n_leafchange = sum(change_counts)
        # correct_changes = val_corrects[changed_indices].sum()
        # entropies_changed = (-val_blogits[changed_indices]*torch.log(val_blogits[changed_indices])).mean()
        # entropies_unchanged = (-val_blogits[changed_indices == False]*torch.log(val_blogits[changed_indices == False])).mean()
        # blogits_changed = val_blogits[changed_indices].max(dim=1)[0].mean()
        # blogits_unchanged = val_blogits[changed_indices == False].max(dim=1)[0].mean()

        # TRAIN, VAL LOG
        logging.info("Epoch: {} | train accs: ({:.4f}, {:.4f}), train losses: ({:.4f}, {:.4f}), valid acc: {:.4f}, valid loss: {:.4f}".format(
            epoch, train_acc, train_inf_acc, train_loss, train_inf_loss, val_acc, val_loss))
        logging.info("LR: {:.6f}, Sigmoid alpha {:.1f}, uncertainity: {:.4f}, leaf distribution {:.4f}".format(
            lr, sigmoid_alpha, train_entropy, train_sample_entropy))
        logging.info("Train | leaf samples: {}\n leaf acc: {}\n leaf targets: {}".format(
            tsdecisions, tleaf_acc, tleaftargets))
        logging.info("Val | leaf samples: {}\n leaf acc: {}\n leaf targets: {}\n".format(
            vsdecisions, vleaf_acc, vleaftargets))
        # logging.info("Val | Changed leaves {}, This many correct: {}, Changed/Unchanged logits: {}/{}, Changed/Unchanged entropies: {}/{}, Changed leaves by depth {}".format(n_leafchange, correct_changes, blogits_changed, blogits_unchanged, entropies_changed, entropies_unchanged, change_counts))
        for log_key in ['train_acc', 'train_inf_acc','train_loss', 'train_inf_loss', 
                        'val_loss', 'val_acc', 'train_entropy', 'train_sample_entropy',
                        'sigmoid_alpha', 'lr']:
            self.metrics[log_key].append(eval(log_key))
            mlflow.log_metric(log_key, eval(log_key), step=epoch)

        # TEST LOG IF IMPROVEMENT
        better_acc, better_loss = (val_acc > best_val[0], val_loss < best_val[1])
        if better_acc or better_loss:
            test_loss, test_acc, _, _, _, _, _ = eval_model_cnn(self.model, self.loader.test, self.device)
            logging.info("TEST LOG | acc: {:.4f}, loss: {:.4f}, \n".format(test_acc, 
                                                                           test_loss))
            if better_acc:
                mlflow.log_metric('test_acc1', test_acc, step=epoch)
                mlflow.log_metric('test_loss1', test_loss, step=epoch)
                best_val[0] = val_acc
            if better_loss:
                mlflow.log_metric('test_acc2', test_acc, step=epoch)
                mlflow.log_metric('test_loss2', test_loss, step=epoch)
                best_val[1] = val_loss
        return best_val, val_leaves

    def step_scheds(self):
        self.sched.step()
        self.sig_sched.step()

    def run(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.setup(cfg.model, cfg.optim, cfg.sched, cfg.sig_sched, cfg.loader, cfg.device)
        self.start_run(cfg.proj_name, cfg.mlflow.username, cfg.mlflow.token, cfg.run_name, 
                       cfg.debug_level, cfg.mlflow.server, cfg.mlflow.exp)

        # Param. logging
        mlflow.log_params({
            'task': cfg.loader.name,
            'epochs': cfg.epochs,
        })
        # Param. logging
        mlflow.log_params(self.get_config())

        # FFF leaf stats
        leafstats = LeafStats(self.model.classifier.n_leaves, cfg.loader.out_dim)

        # Training
        best_val = [0.0, 1e10]
        a = 1.0
        all_leaves = torch.empty(len(cfg.loader.valid.dataset))
        for epoch in range(cfg.epochs):
            if epoch < 0:
                train = train_epoch_warmup(self.model, self.optim, cfg.loader.train, epoch, 
                                           cfg.device)
                self.model.classifier.copy_leaves()
            else:
                train = train_epoch_infa(self.model, self.optim, cfg.loader.train, epoch,
                                         cfg.device, self.sched)
            train_inf = eval_model_cnn(self.model, cfg.loader.train, cfg.device, 
                                       leafstats)
            val = eval_model_cnn(self.model, cfg.loader.valid, cfg.device, 
                                 leafstats)
            self.step_scheds()
            best_val, all_leaves = self.log_epoch(train, train_inf, val, epoch, best_val, cfg.loader.test, 
                                                  all_leaves, cfg.device)

        # Final Test
        test_loss, test_acc, _, _, _, _, _ = eval_model_cnn(self.model, cfg.loader.test, cfg.device)
        logging.info("FINAL TEST | acc: {:.4f}, loss: {:.4f}, ".format(test_acc, test_loss))
        mlflow.log_metrics({
            'test_loss': test_loss,
            'test_acc': test_acc,
        })

        self.end_run(cfg.seed)

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
