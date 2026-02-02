import nibabel as nib
import numpy as np
import os
import pandas as pd
import torch
import torch.optim as optim
from src.scores.ScoreList import ScoreList
from src.utils.EMA import EMA
from src.utils.helper import convert_image, load_json_file, merge_img_label_gt, dump_json_file
from src.utils.helper import merge_img_label_gt_simplified
from src.losses.LossFunctions import DiceLoss
from src.utils.Experiment import Experiment
import seaborn as sns
import math
from matplotlib import figure
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchio as tio
from src.utils.ProjectConfiguration import ProjectConfiguration as pc
from src.utils.vitca_utils import norm_grad
import datetime
from collections import deque

class BaseAgent():
    """Base class for all agents. Handles basic training and only needs to be adapted if special use cases are necessary.
    
    .. note:: In many cases only the data preparation and outputs need to be changed."""
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def set_exp(self, exp: Experiment) -> None:
        r"""Set experiment of agent and initialize.
            #Args
                exp (Experiment): Experiment class"""
        self.exp = exp
        self.initialize()

    def create_optimizer(self, model: torch.nn.Module) -> torch.optim:
        r"""Create optimizer for model
            #Args
                model (torch.nn.Module): model to be optimized
            #Returns:
                optimizer (torch.optim): optimizer
        """
        optimizer_params = {k.replace("trainer.optimizer.", ""): v for k, v in self.exp.config.items() if k.startswith('trainer.optimizer.')}
        optimizer = eval(self.exp.get_from_config('trainer.optimizer'))(model.parameters(), **optimizer_params)
        return optimizer

    def create_scheduler(self, optimizer: torch.optim) -> torch.optim.lr_scheduler:
        scheduler_params = {k.replace("trainer.lr_scheduler.", ""): v for k, v in self.exp.config.items() if k.startswith('trainer.lr_scheduler.')}
        scheduler = eval(self.exp.get_from_config('trainer.lr_scheduler'))(optimizer, **scheduler_params)
        return scheduler

    def initialize(self): 
        r"""Initialize agent with optimizers and schedulers
        """
        self.device = torch.device(self.exp.get_from_config('experiment.device'))
        self.batch_size = self.exp.get_from_config('trainer.batch_size')
        # If stacked NCAs
        if isinstance(self.model, list):
            self.optimizer = []
            self.scheduler = []
            for m in range(len(self.model)):
                self.optimizer.append(self.create_optimizer(self.model[m]))
                    
                self.scheduler.append(self.create_scheduler(self.optimizer[m]))
        else:
            self.optimizer = self.create_optimizer(self.model)
            
            self.scheduler = self.create_scheduler(self.optimizer)

        if self.exp.get_from_config('trainer.find_best_model_on') is not None:
            self.best_model = {
                'epoch': 0,
                'dice': 0
            }

        if self.exp.config['trainer.ema']:
            self.ema: EMA = EMA(self.model, self.exp.config['trainer.ema.decay'])
            assert self.exp.config['trainer.ema.update_per'] in ['batch', 'epoch']

        if self.exp.config.get('trainer.use_amp', False):
             self.scaler = torch.amp.GradScaler('cuda')
        self._init_spike_watch()

    def _get_config(self, key: str, default=None):
        return self.exp.config[key] if key in self.exp.config else default

    def _init_spike_watch(self) -> None:
        self._spike_watch_enabled = bool(self._get_config('experiment.logging.spike_watch.enabled', False))
        self._spike_watch_keys = self._get_config('experiment.logging.spike_watch.keys', [])
        self._spike_watch_window = int(self._get_config('experiment.logging.spike_watch.window', 50))
        self._spike_watch_zscore = float(self._get_config('experiment.logging.spike_watch.zscore', 3.0))
        self._spike_watch_min_value = float(self._get_config('experiment.logging.spike_watch.min_value', 0.0))
        self._spike_watch_max_images_per_epoch = int(self._get_config('experiment.logging.spike_watch.max_images_per_epoch', 10))
        self._spike_watch_max_images_per_spike = int(self._get_config('experiment.logging.spike_watch.max_images_per_spike', 2))
        self._spike_watch_save_classes = self._get_config('experiment.logging.spike_watch.save_classes', [3, 4])
        self._spike_watch_history = {k: deque(maxlen=self._spike_watch_window) for k in self._spike_watch_keys}
        self._spike_watch_epoch_count = 0
        self._current_global_step = None
        self._current_epoch = None
        self._current_batch_index = None

    def _log_batch_class_counts(self, data: dict, loss_ret: dict) -> None:
        labels = data.get('label', None)
        if labels is None:
            return
        with torch.no_grad():
            if labels.dim() not in (4, 5):
                return
            if labels.dim() == 4:
                # BCHW
                counts = labels.sum(dim=(0, 2, 3))
                total = labels.sum()
            else:
                # B C H W D
                counts = labels.sum(dim=(0, 2, 3, 4))
                total = labels.sum()
            for c in range(counts.shape[0]):
                loss_ret[f"batch_class_pixels/{c}"] = counts[c].item()
                loss_ret[f"batch_class_frac/{c}"] = (counts[c] / (total + 1e-8)).item()

    def _maybe_log_spike_batch(self, data: dict, out: dict, loss_ret: dict) -> None:
        if not self._spike_watch_enabled or not self._spike_watch_keys:
            return
        if self._current_global_step is None:
            return

        spike_keys = []
        for key in self._spike_watch_keys:
            if key not in loss_ret:
                continue
            value = loss_ret[key]
            if isinstance(value, torch.Tensor):
                value = value.item()
            history = self._spike_watch_history.setdefault(key, deque(maxlen=self._spike_watch_window))
            if len(history) >= 5:
                mean = float(np.mean(history))
                std = float(np.std(history))
                threshold = max(self._spike_watch_min_value, mean + self._spike_watch_zscore * std)
                if std > 0 and value > threshold:
                    spike_keys.append((key, value, threshold))
            history.append(value)

        if not spike_keys:
            return
        if self._spike_watch_epoch_count >= self._spike_watch_max_images_per_epoch:
            return

        with torch.no_grad():
            logits = out.get('logits', None)
            if logits is None:
                logits = out.get('probabilities', None)
            if logits is None:
                return

            # logits: BHWC, labels: BCHW
            labels = data.get('label', None)
            if labels is None:
                return

            image = data.get('image', None)
            if image is None:
                return

            if image.dim() == 4:
                image_bhwc = image.permute(0, 2, 3, 1)
            else:
                return

            gt_bhwc = labels.permute(0, 2, 3, 1)
            pred_bhwc = logits

            class_idx = [c for c in self._spike_watch_save_classes if c < pred_bhwc.shape[-1]]
            if len(class_idx) == 0:
                return

            pred_sel = pred_bhwc[..., class_idx]
            gt_sel = gt_bhwc[..., class_idx]

            is_rgb = False
            try:
                is_rgb = self.exp.datasets['train'].is_rgb
            except Exception:
                pass

            max_imgs = min(self._spike_watch_max_images_per_spike, image_bhwc.shape[0])
            for bi in range(max_imgs):
                try:
                    merged = merge_img_label_gt_simplified(
                        image_bhwc[bi:bi+1],
                        pred_sel[bi:bi+1],
                        gt_sel[bi:bi+1],
                        rgb=is_rgb
                    )
                    for key, value, threshold in spike_keys:
                        tag = f"spike/{key}/epoch_{self._current_epoch}/step_{self._current_global_step}_b{bi}"
                        self.exp.write_img(tag, merged, self._current_global_step)
                except Exception as e:
                    print(f"[spike_watch] Skipped visualization for batch {self._current_global_step} (b{bi}): {e}")
            self._spike_watch_epoch_count += max_imgs

    def printIntermediateResults(self, loss: torch.Tensor, epoch: int) -> None:
        r"""Prints intermediate results of training and adds it to tensorboard
            #Args 
                loss (torch)
                epoch (int) 
        """
        print(epoch, "loss =", loss.item())
        self.exp.save_model()


    def prepare_data(self, data: list, eval: bool = False) -> list:
        r"""If any data preparation needs to be done do it here. 
            #Args
                data ([]): The data to be processed.
                eval (Bool): Whether or not its for evaluation. 
        """
        return data

    def get_outputs(self, data: torch.Tensor, **kwargs) -> dict:
        r"""Get the output of the model.
            #Args 
                data (torch): The data to be passed to the model.
        """
        return self.model(data)

    def initialize_epoch(self) -> None:
        r"""Everything that should happen once before each epoch should be defined here.
        """
        return

    def conclude_epoch(self) -> None:
        r"""Everything that should happen once after each epoch should be defined here.
        """
        return

    def batch_step(self, data: tuple, loss_f: torch.nn.Module) -> dict:
        r"""Execute a single batch training step
            #Args
                data (tensor, tensor): inputs, targets
                loss_f (torch.nn.Module): loss function
            #Returns:
                loss item
        """
        data = self.prepare_data(data)
        # data["image"]: BCHW
        # data["label"]: BCHW
        out = self.get_outputs(data)
        self.optimizer.zero_grad()
        loss = 0
        loss_ret = {}
        #print(outputs.shape, targets.shape)
        #2D: outputs: BHWC, targets: BHWC
        out["target_unpatched"] = data["label"]
        if "label_dist" in data:
            out["target_dist"] = data["label_dist"]
        if self.exp.config.get('trainer.use_amp', False):
            # Using AMP
            with torch.amp.autocast('cuda'):
                loss, loss_ret = loss_f(**out)
        else:
            loss, loss_ret = loss_f(**out)

        self._log_batch_class_counts(data, loss_ret)
        self._maybe_log_spike_batch(data, out, loss_ret)

        do_backward = False
        if isinstance(loss, torch.Tensor):
            if loss.numel() > 1:
                loss = loss.mean()
            # If loss was 6 elements, .mean() reduces it to a scalar tensor (0-dim or 1-dim size 1)
            # calling .item() should now work.
            if loss.item() != 0:
                do_backward = True
        elif loss != 0:
            do_backward = True

        if do_backward:
            if self.exp.config.get('trainer.use_amp', False):
                self.scaler.scale(loss).backward()
                if self.exp.config['trainer.normalize_gradients'] == "all" or self.exp.config['experiment.logging.track_gradient_norm']:
                     self.scaler.unscale_(self.optimizer)
                     # ... unscaling happens
            else:
                loss.backward()

            if self.exp.config['trainer.normalize_gradients'] == "all" or self.exp.config['experiment.logging.track_gradient_norm']:
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

            if self.exp.config['trainer.normalize_gradients'] == "all":
                max_norm = 1.0
                # Gradient normalization

                # Calculate scaling factor and scale gradients if necessary
                scale_factor = max_norm / (total_norm + 1e-6)
                if scale_factor < 1:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.data.mul_(scale_factor)
            elif self.exp.config['trainer.normalize_gradients'] == "layerwise":
                with torch.no_grad():
                    norm_grad(self.model)

            if self.exp.config['experiment.logging.track_gradient_norm']:
                if not hasattr(self, 'epoch_grad_norm'):
                    self.epoch_grad_norm = []
                self.epoch_grad_norm.append(total_norm)

            if self.exp.config.get('trainer.use_amp', False):
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if not self.exp.config['trainer.update_lr_per_epoch']:
                self.update_lr()
            
            if self.exp.config['trainer.ema'] and self.exp.config['trainer.ema.update_per'] == 'batch':
                self.ema.update()

        return loss_ret

    def update_lr(self) -> None:
        for i, lr in enumerate(self.scheduler.get_last_lr()):
            self.exp.write_scalar(f'lr/{i}', lr, self.scheduler.last_epoch)
        self.scheduler.step()
        

    def intermediate_results(self, epoch: int, loss_log: list) -> None:
        r"""Write intermediate results to tensorboard
            #Args
                epoch (int): Current epoch
                los_log ([loss]): Array of losses
        """
        for key in loss_log.keys():
            if len(loss_log[key]) != 0:
                average_loss = sum(loss_log[key]) / len(loss_log[key])
            else:
                average_loss = 0
            print(f"Loss/train/{key} = {average_loss}")
            self.exp.write_scalar('Loss/train/' + str(key), average_loss, epoch)

    def plot_results_byPatient(self, loss_log: dict) -> figure:
        r"""Plot losses in a per patient fashion with seaborn to display in tensorboard.
            #Args
                loss_log ({name: loss}: Dictionary of losses
        """
        print(loss_log)
        sns.set_theme()
        plot = sns.scatterplot(x=loss_log.keys(), y=loss_log.values())
        plot.set(ylim=(0, 1))
        plot = plot.get_figure()
        return plot

    def intermediate_evaluation(self, epoch: int, split='test') -> None:
        r"""Do an intermediate evluation during training 
            .. todo:: Make variable for more evaluation scores (Maybe pass list of metrics)
            #Args
                dataset (Dataset)
                epoch (int)
        """
        if self.exp.get_from_config('trainer.ema'):
            self.ema.apply_shadow()
        
        scores = ScoreList(self.exp.config)
        loss_log = self.test(scores, split=split, tag=f'{split}/img/')
        if self.exp.get_from_config('trainer.ema'):
            self.ema.restore_original()

        
        if self.exp.get_from_config('trainer.datagen.difficulty_weighted_sampling') and False:
            raise NotImplementedError("This must be updated")
            # This will not be that easy! The SlimDataLoader computes weighted_difficulties once. They are not updated during training!!!
            assert loss_log is not None
            loss_sum_per_patient = {}
            for mask in loss_log.keys():
                for patient_id in loss_log[mask].keys():
                    if patient_id not in loss_sum_per_patient:
                        loss_sum_per_patient[patient_id] = 0
                    loss_sum_per_patient[patient_id] += loss_log[mask][patient_id]
            for patient_id in loss_sum_per_patient.keys():
                loss_sum_per_patient[patient_id] /= len(loss_log.keys())
                #loss_log does not contain the loss but the segmentation score!
                self.exp.dataset.difficulties[patient_id] = 1 - loss_sum_per_patient[patient_id]
            print(f"Updated difficulties for {len(loss_sum_per_patient)} patients")
            self.exp.dataset.dataloader.restart()

        if loss_log is not None:
            for key in loss_log.keys():
                img_plot = self.plot_results_byPatient(loss_log[key])
                self.exp.write_figure(f'Patient/{split}/' + str(key), img_plot, epoch)
                if len(loss_log[key]) > 0:
                    self.exp.write_scalar(f'{split}/' + str(key), sum(loss_log[key].values())/len(loss_log[key]), epoch)
                    self.exp.write_histogram(f'{split}/byPatient/' + str(key), np.fromiter(loss_log[key].values(), dtype=float), epoch)
        param_lst = []

        if self.exp.get_from_config('trainer.find_best_model_on') == split:
            assert loss_log is not None
            dices = [sum(loss_log[key].values())/len(loss_log[key]) for key in loss_log.keys()]
            dice = sum(dices) / len(dices)
            if dice > self.best_model['dice']:
                self.best_model['dice'] = dice
                self.best_model['epoch'] = epoch
                self.exp.write_scalar('Model/best_dice', dice, epoch)



        # TODO: ADD AGAIN 
        #for param in self.model.parameters():
        #    param_lst.extend(np.fromiter(param.flatten(), dtype=float))
        #self.exp.write_histogram('Model/weights', np.fromiter(param_lst, dtype=float), epoch)

    def getAverageDiceScore(self, pseudo_ensemble: bool = False, ood_augmentation: tio.Transform=None, output_name: str=None,
                            export_prediction: bool=False) -> dict:
        r"""Get the average Dice test score.
            #Returns:
                return (float): Average Dice score of test set. """
        
        if self.exp.get_from_config("trainer.find_best_model_on") is not None:
            current_params = {k: v.cpu() for k, v in self.model.state_dict().items()}
            find_best_model_on = self.exp.get_from_config("trainer.find_best_model_on")
            best_model_epoch = self.best_model['epoch']
            print(f"Loading best model that was found during training on the {find_best_model_on} set, which is from epoch {best_model_epoch}")
            self.load_state(os.path.join(pc.FILER_BASE_PATH, self.exp.config['experiment.model_path'], 'models', 'epoch_' + str(self.best_model['epoch'])), pretrained=True)
        elif self.exp.get_from_config("trainer.ema"):
            self.ema.apply_shadow()


        scores = ScoreList(self.exp.config)
        loss_log = self.test(scores, save_img=None, pseudo_ensemble=pseudo_ensemble, ood_augmentation=ood_augmentation,
                             output_name=output_name,
                             export_prediction=export_prediction)

        #loss_log = self.test(scores, save_img=None, pseudo_ensemble=pseudo_ensemble, ood_augmentation=ood_augmentation,
        #                     output_name=output_name,
        #                     export_prediction=export_prediction,
        #                     split='train', prediction_export_path="temp")
        #assert False, "remove those changes"

        if self.exp.get_from_config("trainer.find_best_model_on") is not None:
            self.model.load_state_dict(current_params)
        elif self.exp.get_from_config("trainer.ema"):
            self.ema.restore_original()


        eval_file = os.path.join(pc.FILER_BASE_PATH, self.exp.get_from_config("experiment.model_path"), "eval", "standard.csv")
        if ood_augmentation is None and not os.path.exists(eval_file):
            print("write results to", eval_file)
            os.makedirs(os.path.dirname(eval_file), exist_ok=True)
            df = pd.DataFrame(loss_log)
            df.to_csv(eval_file, sep='\t')
        elif ood_augmentation is not None and output_name is not None \
                and not os.path.exists(os.path.join(pc.FILER_BASE_PATH, self.exp.get_from_config("experiment.model_path"), "eval", output_name)):
            eval_file = os.path.join(pc.FILER_BASE_PATH, self.exp.get_from_config("experiment.model_path"), "eval", output_name)
            print("write results to", eval_file)
            os.makedirs(os.path.dirname(eval_file), exist_ok=True)
            df = pd.DataFrame(loss_log)
            df.to_csv(eval_file, sep='\t')


        return loss_log

    def predictOnPath(self, path: str, useSigmoid: bool = True, tag: str = "", pseudo_ensemble: bool = False) -> dict:
        r"""Get the average Dice test score.
            #Returns:
                return (float): Average Dice score of test set. """
        loss_log = self.test(ScoreList(self.exp.config), save_img=[], pseudo_ensemble=pseudo_ensemble)

        return loss_log

    def save_state(self, model_path: str) -> None:
        r"""Save state of current model
        """
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pth'))
        torch.save(self.scheduler.state_dict(), os.path.join(model_path, 'scheduler.pth'))
        if self.exp.get_from_config('trainer.find_best_model_on') is not None:
            dump_json_file(self.best_model, os.path.join(pc.FILER_BASE_PATH, self.exp.config['experiment.model_path'], 'best_model.json'))
        
        if self.exp.get_from_config('trainer.ema'):
            torch.save(self.ema.shadow, os.path.join(model_path, 'ema.pth'))

    def load_state(self, model_path: str, pretrained=False) -> None:
        r"""Load state of current model
        """
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
        if not pretrained:
            self.optimizer.load_state_dict(torch.load(os.path.join(model_path, 'optimizer.pth')))
            self.scheduler.load_state_dict(torch.load(os.path.join(model_path, 'scheduler.pth')))
            if self.exp.get_from_config('trainer.find_best_model_on') is not None:
                self.best_model = load_json_file(os.path.join(pc.FILER_BASE_PATH, self.exp.config['experiment.model_path'], 'best_model.json'))
        
        if self.exp.get_from_config('trainer.ema'):
            loaded_shadow = torch.load(os.path.join(model_path, 'ema.pth'))
            assert self.ema.shadow.keys() == loaded_shadow.keys(), "Loaded EMA shadow does not match model parameters!"
            self.ema.shadow = loaded_shadow

    def train(self, dataloader: DataLoader, loss_f: torch.Tensor) -> None:
        r"""Execute training of model
            #Args
                dataloader (Dataloader): contains training data
                loss_f (nn.Model): The loss for training"""
        for epoch in range(self.exp.currentStep, self.exp.get_max_steps()+1):
            torch.cuda.reset_peak_memory_stats(self.device)
            print(f"{datetime.datetime.now().strftime('%I:%M%p, %B %d, %Y')} Epoch: {epoch}")
            if epoch == self.exp.currentStep:
                self.exp.watch_model(self.model) # Watch model at start of training
            self.exp.set_model_state('train')
            loss_log = {}
            if getattr(self, '_spike_watch_enabled', False):
                self._spike_watch_epoch_count = 0
            self.initialize_epoch()
            print('Dataset size: ' + str(len(dataloader)))
            for i, data in enumerate(tqdm(dataloader)):
                self._current_global_step = (epoch * len(dataloader)) + i
                self._current_epoch = epoch
                self._current_batch_index = i
                loss_item = self.batch_step(data, loss_f)
                
                # Step-wise logging
                global_step = self._current_global_step
                for key, value in loss_item.items():
                    if key.startswith("batch_class_pixels/") or key.startswith("batch_class_frac/"):
                        val = value.item() if isinstance(value, torch.Tensor) else value
                        self.exp.write_scalar(f'Loss/train_step/{key}', val, global_step)
                if i % 10 == 0:
                    for key, value in loss_item.items():
                        if isinstance(value, torch.Tensor):
                            val = value.item()
                        else:
                            val = value
                        self.exp.write_scalar(f'Loss/train_step/{key}', val, global_step)

                for key in loss_item.keys():
                    if key not in loss_log:
                        loss_log[key] = []
                    if isinstance(loss_item[key], float):
                        loss_log[key].append(loss_item[key])
                    else:
                        loss_log[key].append(loss_item[key].detach())

            if epoch == 2:
                print("measure memory allocation")
                mem_allocation = torch.cuda.max_memory_allocated(self.device)
                mem_allocation_mb = mem_allocation/ 1024**2
                print(f"allocated memory: {mem_allocation_mb} MiB")
                mem_allocation_dict ={
                    'byte': mem_allocation,
                    'MiB': mem_allocation_mb
                }
                dump_json_file(mem_allocation_dict, os.path.join(pc.FILER_BASE_PATH, self.exp.config['experiment.model_path'], 'mem_allocation.json'))

                num_params = sum(p.numel() for p in self.model.parameters())
                num_params_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"model has {num_params} params, of which {num_params_trainable} are trainable")
                num_params_dict ={
                    'num_params': num_params,
                    'num_params_trainable': num_params_trainable
                }
                dump_json_file(num_params_dict, os.path.join(pc.FILER_BASE_PATH, self.exp.config['experiment.model_path'], 'num_params.json'))

            if self.exp.get_from_config('trainer.ema') and self.exp.get_from_config('trainer.ema.update_per') == 'epoch':
                self.ema.update()

            self.maybe_track_grad_norm()
            if self.exp.get_from_config('trainer.update_lr_per_epoch'):
                self.update_lr()
            self.intermediate_results(epoch, loss_log)
            do_eval = False
            if self.exp.get_from_config('trainer.always_eval_in_last_epochs') is not None:
                if epoch > self.exp.get_max_steps() - self.exp.get_from_config('trainer.always_eval_in_last_epochs'):
                    do_eval = True 
            if epoch % self.exp.get_from_config('experiment.logging.evaluate_interval') == 0 or do_eval:
                print("Evaluate model")
                self.intermediate_evaluation(epoch, split='test')
                if self.exp.get_from_config('experiment.logging.also_eval_on_train'):
                    self.intermediate_evaluation(epoch, split='train')
            #if epoch % self.exp.get_from_config('ood_interval') == 0:
            #    print("Evaluate model in OOD cases")
            #    self.ood_evaluation(epoch=epoch)
            save_best_model = False
            if self.exp.get_from_config('trainer.find_best_model_on') is not None:
                if self.best_model['epoch'] == epoch:
                    save_best_model = True
            if epoch % self.exp.get_from_config('experiment.save_interval') == 0 or save_best_model:
                print("Model saved")
                self.save_state(os.path.join(pc.FILER_BASE_PATH, self.exp.get_from_config('experiment.model_path'), 'models', 'epoch_' + str(self.exp.currentStep)))
            self.conclude_epoch()
            self.exp.increase_epoch()

    def prepare_image_for_display(self, image: torch.Tensor) -> torch.Tensor:
        r"""Prepare an image to be displayed in tensorboard. Since images need to be in a specific format these modifications these can be done here.
            #Args
                image (torch): The image to be processed for display. 
        """
        return image

    def maybe_track_grad_norm(self) -> None:
        if not self.exp.get_from_config('experiment.logging.track_gradient_norm'):
            return
        self.exp.write_scalar('Model/grad_norm', np.mean(self.epoch_grad_norm), self.exp.currentStep)
        self.epoch_grad_norm = []

    #def ood_evaluation(self, ood_cases=["random_noise", "random_spike", "random_anitrosopy"], epoch=0):
    #    print("OOD EVALUATION")
    #    dataset_train = self.exp.dataset
    #    diceLoss = DiceLoss(useSigmoid=True)
    #    for augmentation in ood_cases:
    #        dataset_eval = Nii_Gz_Dataset(aug_type=augmentation)
    #        self.exp.dataset = dataset_eval
    #        loss_log = self.test(diceLoss, tag='ood/' + str(augmentation) + '/')
    #        for key in loss_log.keys():
    #            self.exp.write_scalar('ood/Dice/' + str(key) + ", " + str(augmentation), sum(loss_log[key].values())/len(loss_log[key]), epoch)
    #            self.exp.write_histogram('ood/Dice/' + str(key) + ", " + str(augmentation) + '/byPatient', np.fromiter(loss_log[key].values(), dtype=float), epoch)
    #    self.exp.dataset = dataset_train

    
    def compute_nqm_score(self, prediction_stack: np.ndarray) -> float:
        mean = np.sum(prediction_stack, axis=0) / prediction_stack.shape[0]
        stdd = 0
        for i in range(prediction_stack.shape[0]):
            img = prediction_stack[i] - mean
            img = np.power(img, 2)
            stdd = stdd + img
        stdd = stdd / prediction_stack.shape[0]
        stdd = np.sqrt(stdd)
        return np.sum(stdd) / np.sum(mean)

    def labelVariance(self, images: torch.Tensor, median: torch.Tensor, img_mri: torch.Tensor, img_id: str, targets: torch.Tensor) -> None:
        r"""Calculate variance over all predictions
            #Args
                images (torch): The inferences
                median: The median of all inferences
                img_mri: The mri image
                img_id: The id of the image
                targets: The target segmentation
        """
        mean = np.sum(images, axis=0) / images.shape[0]
        stdd = 0
        for id in range(images.shape[0]):
            img = images[id] - mean
            img = np.power(img, 2)
            stdd = stdd + img
        stdd = stdd / images.shape[0]
        stdd = np.sqrt(stdd)

        print("NQM Score: ", np.sum(stdd) / np.sum(median))

        return

    @staticmethod
    def standard_deviation(loss_log: dict) -> float:
        r"""Calculate the standard deviation
            #Args
                loss_log: losses
        """
        mean = sum(loss_log.values())/len(loss_log)
        stdd = 0
        for e in loss_log.values():
            stdd = stdd + pow(e - mean, 2)
        stdd = stdd / len(loss_log)
        stdd = math.sqrt(stdd)
        return stdd
    
    
    @torch.no_grad()
    def test(self, loss_f: torch.nn.Module, save_img: list = None, tag: str = 'test/img/', 
             pseudo_ensemble: bool = False, split='test',
             ood_augmentation: tio.Transform=None,
             output_name: str=None,
             prediction_export_path: str="pred") -> dict:
        raise NotImplementedError
