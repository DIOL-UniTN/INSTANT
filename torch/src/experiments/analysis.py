import os 
import torch
import hydra
import dagshub
import mlflow
import logging
from pathlib import Path
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, mlflow_id, exp_conf: int):
        self.exp_name = "Analysis"
        self.mlflow_id = mlflow_id
        self.draw_limit = 8
        self.certainity_th = 1.483e-05

        self.exp_conf = exp_conf

    def setup(self, exps, partial_model, loader, device):
        depth = partial_model.keywords['depth']
        state_dict_path = mlflow.artifacts.download_artifacts(
                run_id=exps[self.exp_conf][depth - 1], 
                artifact_path="state_dict.pt", dst_path=self.out_dir)

        # Model and optim.setup
        self.model = partial_model(in_features=loader.in_chan*loader.in_size[0]*loader.in_size[1], 
                           out_features=loader.out_dim).to(device)
        self.model.load_state_dict(torch.load(state_dict_path))
        self.criterion = torch.nn.CrossEntropyLoss()

    def start_run(self, proj_name, username, mlflow_pass, run_name, debug_level:int):
        # MLFlow setup
        self.out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.log_filename = hydra.core.hydra_config.HydraConfig.get().job.name+'.log'
        self.overrides_config = self.out_dir/'.hydra/overrides.yaml'

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

    def end_run(self, seed:int):
        logging.info("Success!")
        mlflow.log_param('success', True)

        # Log base
        mlflow.log_param('seed', seed)
        mlflow.log_artifact(self.out_dir/self.log_filename)
        mlflow.log_artifact(self.overrides_config)

        mlflow.end_run()
        finished_run = mlflow.get_run(self.run.info.run_id)
        logging.info(f"MLFlow run ID: {finished_run.info.run_id}, status: {finished_run.info.status}")

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

    def draw_imgs(self, imgs: torch.Tensor, target: int, title: str, preds: list) -> None:
        return
        imgs = imgs[:self.draw_limit]
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        fig.suptitle(preds[:self.draw_limit], fontsize=16, fontweight='bold')
        for i, ax in enumerate(axes.flat):
            if i > imgs.size(0) - 1:
                break
            img = imgs[i].cpu().permute(1, 2, 0).numpy()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(self.out_dir/f"{title}{target}.png", dpi=300)
        mlflow.log_artifact(self.out_dir/f"{title}{target}.png")
        plt.close()

    def run(self, cfg):
        logging.info(f"Running {self.exp_name} with seed: {cfg.seed}")
        self.start_run(cfg.proj_name, cfg.username, cfg.mlflow_pass, 
                       cfg.run_name, cfg.debug_level)
        self.setup(cfg.mlflow_exp, cfg.model, cfg.loader,
                   cfg.device)


        # Explain
        self.model.eval()
        certain_corrects, uncertain_corrects = 0, 0
        certains, uncertains = 0, 0
        corrects = 0
        samples = 0
        eval_certain_corrects, eval_uncertain_corrects = 0, 0
        eval_certains, eval_uncertains = 0, 0
        eval_corrects = 0
        eval_samples = 0
        for i, (inputs, targets) in enumerate(cfg.loader.test):
            print(f"{i}-th iteration.")
            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            eval_inputs, eval_targets = inputs, targets
            outputs, entropies, sample_entropies = self.model.train_forward(inputs)
            eval_outputs, _ = self.model.eval_forward(inputs)

            _, preds = torch.max(outputs.data, 1)
            corrects += (preds == targets).sum()
            samples += inputs.size(0)
            mean_entropies = entropies.mean(dim=1)
            print(f"meaner entropies: {mean_entropies.mean()}")

            # # TODO: Draw cer-acc and also layerwise for paper one day lmao sadge?
            # draw_certainity_acc(mean_entropies, )

            
            # breakpoint()
            uncertain_entropies = entropies[mean_entropies > self.certainity_th]
            certain_entropies = entropies[mean_entropies < self.certainity_th]
            # print(certain_entropies.max().item())
            # print(certain_entropies.min().item())
            # print(certain_entropies.mean())
            # print(uncertain_entropies.min().item())
            # breakpoint()
            uncertain_inputs, uncertain_targets, uncertain_preds = (inputs[mean_entropies  > self.certainity_th], 
                                                                    targets[mean_entropies > self.certainity_th],
                                                                    preds[mean_entropies > self.certainity_th])
            uncertain_corrects += (uncertain_targets == uncertain_preds).sum()
            uncertains +=  uncertain_targets.size(0)
            certain_inputs, certain_targets, certain_preds = (inputs[mean_entropies  < self.certainity_th], 
                                                              targets[mean_entropies < self.certainity_th],
                                                              preds[mean_entropies < self.certainity_th])
            certain_corrects += (certain_targets == certain_preds).sum()
            certains +=  certain_targets.size(0)

            # EVAL
            _, eval_preds = torch.max(eval_outputs.data, 1)
            eval_corrects += (eval_preds == targets).sum()
            eval_samples += eval_inputs.size(0)

            eval_uncertain_inputs, eval_uncertain_targets, eval_uncertain_preds = (inputs[mean_entropies  > self.certainity_th], 
                                                                    targets[mean_entropies > self.certainity_th],
                                                                    eval_preds[mean_entropies > self.certainity_th])
            eval_certain_inputs, eval_certain_targets, eval_certain_preds = (inputs[mean_entropies  < self.certainity_th], 
                                                              targets[mean_entropies < self.certainity_th],
                                                              eval_preds[mean_entropies < self.certainity_th])
            # TODO: Analyse depthwise certainity
            eval_uncertain_corrects += (eval_uncertain_targets == eval_uncertain_preds).sum()
            eval_certain_corrects += (eval_certain_targets == eval_certain_preds).sum()
            eval_uncertain_inputs, eval_uncertain_targets, eval_uncertain_preds = (eval_inputs[mean_entropies  > self.certainity_th], 
                                                                                   eval_targets[mean_entropies > self.certainity_th],
                                                                                   eval_preds[mean_entropies > self.certainity_th])
            eval_certain_inputs, eval_certain_targets, eval_certain_preds = (eval_inputs[mean_entropies  < self.certainity_th], 
                                                                             eval_targets[mean_entropies < self.certainity_th],
                                                                             eval_preds[mean_entropies < self.certainity_th])
            eval_certains +=  eval_certain_targets.size(0)
            eval_uncertains += eval_uncertain_targets.size(0)
            if i == 0:
                for target in range(self.model.out_features):
                    imgs = uncertain_inputs[uncertain_targets == target]
                    preds = uncertain_preds[uncertain_targets == target]
                    self.draw_imgs(imgs, target, 'uncertain', preds.tolist())
                    imgs = certain_inputs[certain_targets == target]
                    preds = certain_preds[certain_targets == target]
                    self.draw_imgs(imgs, target, 'certain', preds.tolist())

                for target in range(self.model.out_features):
                    imgs = eval_uncertain_inputs[eval_uncertain_targets == target]
                    eval_preds = eval_uncertain_preds[eval_uncertain_targets == target]
                    self.draw_imgs(imgs, target, 'eval_uncertain', eval_preds.tolist())
                    imgs = eval_certain_inputs[eval_certain_targets == target]
                    eval_preds = certain_preds[eval_certain_targets == target]
                    self.draw_imgs(imgs, target, 'eval_certain', eval_preds.tolist())

        print(certain_entropies.mean(dim=1), uncertain_entropies.mean(dim=1))
        logging.info(f"Train mode TEST acc: {corrects/samples}, Certain acc: {certain_corrects/certains}, Uncertain acc: {uncertain_corrects/uncertains}")
        logging.info(f"Eval mode TEST acc: {eval_corrects/eval_samples}, Certain acc: {eval_certain_corrects/eval_certains}, Uncertain acc: {eval_uncertain_corrects/eval_uncertains}")
        logging.info(f"Eval mode TEST, certains: {eval_certains}, uncertains: {eval_uncertains}")
        logging.info(f"Eval mode TEST, certain entropies: {certain_entropies.mean()}, depthwise: {certain_entropies.mean(dim=0)}")
        logging.info(f"Eval mode TEST, uncertain entropies: {uncertain_entropies.mean()}, depthwise: {uncertain_entropies.mean(dim=0)}")
        # Param. logging
        mlflow.log_params({
            'task': cfg.loader.name,
            })
        # Model param. logging
        mlflow.log_params(self.model.get_config())
        # Final Test
        self.end_run(cfg.seed)

    def draw_certainity_acc(self, entropies, corrects):
        pass

    def main(self, cfg):
        try: 
            self.run(cfg)
        except Exception as e:
            self.end_failure(e, cfg.seed)
