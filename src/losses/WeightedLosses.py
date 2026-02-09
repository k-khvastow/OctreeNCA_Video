import einops
import torch
import torch.nn as nn
import src.losses.DiceBCELoss
import src.losses.OverflowLoss
import src.losses.MaskedL1Loss
import src.losses.IntermediateSupervision
import src.losses.BCELoss
import src.losses.DiceLoss
import src.losses.LossFunctions


class WeightedLosses(nn.Module):
    def __init__(self, config):
        super(WeightedLosses, self).__init__()
        assert len(config['trainer.losses']) == len(config['trainer.loss_weights']), f"{config['trainer.losses']} and {config['trainer.loss_weights']} must have the same length"
        self.losses = []
        self.weights = []
        for i, _ in enumerate(config['trainer.losses']):
            try:
                self.losses.append(eval(config['trainer.losses'][i])(config=config))
            except TypeError:
                try:
                    loss_parameters = config['trainer.losses.parameters'][i]
                    self.losses.append(eval(config['trainer.losses'][i])(**loss_parameters))
                except TypeError:
                    self.losses.append(eval(config['trainer.losses'][i])())
            
            self.weights.append(config['trainer.loss_weights'][i])
            
            

    def forward(self, **kwargs):
        shared_kwargs = dict(kwargs)
        logits = kwargs.get("logits", None)
        target = kwargs.get("target", None)
        probabilities = kwargs.get("probabilities", None)
        target_dist = kwargs.get("target_dist", None)

        logits_cf = None
        if logits is not None:
            if logits.dim() == 4 and logits.shape[-1] < logits.shape[1]:
                logits_cf = einops.rearrange(logits, "b h w c -> b c h w")
            elif logits.dim() == 5 and logits.shape[-1] < logits.shape[1]:
                logits_cf = einops.rearrange(logits, "b h w d c -> b c h w d")
            else:
                logits_cf = logits
            shared_kwargs["logits_cf"] = logits_cf

        if target is not None and logits_cf is not None:
            if target.dim() == logits_cf.dim() and target.shape[1] == logits_cf.shape[1]:
                target_cf = target
            elif target.dim() == 4 and target.shape[-1] == logits_cf.shape[1] and target.shape[-1] < target.shape[1]:
                target_cf = einops.rearrange(target, "b h w c -> b c h w")
            elif target.dim() == 5 and target.shape[-1] == logits_cf.shape[1] and target.shape[-1] < target.shape[1]:
                target_cf = einops.rearrange(target, "b h w d c -> b c h w d")
            else:
                target_cf = target
            shared_kwargs["target_cf"] = target_cf

            if target_cf.dim() == logits_cf.dim() and target_cf.shape[1] == logits_cf.shape[1] and torch.is_floating_point(target_cf):
                shared_kwargs["target_idx"] = torch.argmax(target_cf, dim=1)
            elif target_cf.dim() == logits_cf.dim() and target_cf.shape[1] == 1:
                shared_kwargs["target_idx"] = target_cf[:, 0].long()
            elif target_cf.dim() == logits_cf.dim() - 1:
                shared_kwargs["target_idx"] = target_cf.long()

        if probabilities is not None:
            if probabilities.dim() == 4 and probabilities.shape[-1] < probabilities.shape[1]:
                shared_kwargs["probabilities_cf"] = einops.rearrange(probabilities, "b h w c -> b c h w")
            elif probabilities.dim() == 5 and probabilities.shape[-1] < probabilities.shape[1]:
                shared_kwargs["probabilities_cf"] = einops.rearrange(probabilities, "b h w d c -> b c h w d")
            else:
                shared_kwargs["probabilities_cf"] = probabilities

        if target_dist is not None:
            if target_dist.dim() == 4 and target_dist.shape[-1] < target_dist.shape[1]:
                shared_kwargs["target_dist_cf"] = einops.rearrange(target_dist, "b h w c -> b c h w")
            elif target_dist.dim() == 5 and target_dist.shape[-1] < target_dist.shape[1]:
                shared_kwargs["target_dist_cf"] = einops.rearrange(target_dist, "b h w d c -> b c h w d")
            else:
                shared_kwargs["target_dist_cf"] = target_dist

        loss = 0
        loss_ret = {}
        for i, _ in enumerate(self.losses):
            e = None
            try:
                r = self.losses[i](**shared_kwargs)
            except TypeError as e:
                try:
                    r = self.losses[i](**kwargs)
                    e = None
                except TypeError:
                    pass

            if isinstance(e, TypeError):
                if logits_cf is not None and "target_cf" in shared_kwargs:
                    logits_local = logits_cf
                    target_local = shared_kwargs["target_cf"]
                else:
                    if kwargs["logits"].dim() == 5:
                        logits_local = einops.rearrange(kwargs["logits"], "b h w d c -> b c h w d")
                        target_local = einops.rearrange(kwargs["target"], "b h w d c -> b c h w d")
                    else:
                        assert kwargs["logits"].dim() == 4, f"Expected 4D tensor, got {kwargs['logits'].dim()}"
                        logits_local = einops.rearrange(kwargs["logits"], "b h w c -> b c h w")
                        target_local = einops.rearrange(kwargs["target"], "b h w c -> b c h w")
                r = self.losses[i](logits_local, target_local)
            
            if isinstance(r, tuple):
                l, d = r
            else:
                l = r
                d = {'loss': l.item()}
            loss += l * self.weights[i]
            for k, v in d.items():
                loss_ret[f"{self.losses[i].__class__.__name__}/{k}"] = d[k] * self.weights[i]
        return loss, loss_ret
