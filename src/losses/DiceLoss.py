import einops
import torch
import numpy as np
import torch.nn.functional as F
from src.utils.DistanceMaps import batch_signed_distance_map

class DiceLoss(torch.nn.Module):
    def __init__(self, useSigmoid: bool = True, smooth: float = 1) -> None:
        self.useSigmoid = useSigmoid
        self.smooth = smooth
        super().__init__()

    def compute(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.useSigmoid:
            probabilities = torch.sigmoid(logits)
        probabilities = torch.flatten(probabilities) 
        target = torch.flatten(target)
        
        intersection = (probabilities * target).sum()
        dice_loss = 1 - (2.*intersection + self.smooth)/(probabilities.sum() + target.sum() + self.smooth)
        
        return dice_loss
    

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs):
        assert logits.shape == target.shape
        loss_ret = {}
        loss = 0
        if len(logits.shape) == 5 and target.shape[-1] == 1:
            for m in range(target.shape[-1]):
                loss_loc = self.compute(logits[..., m], target[...])
                loss = loss + loss_loc
                loss_ret[f"mask_{m}"] = loss_loc.item()
        else:
            for m in range(target.shape[-1]):
                if 1 in target[..., m]:
                    loss_loc = self.compute(logits[..., m], target[..., m])
                    loss = loss + loss_loc
                    loss_ret[f"mask_{m}"] = loss_loc.item()

        return loss, loss_ret
    


## TAKEN from nnUNet

def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp
def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn
class nnUNetSoftDiceLoss(torch.nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        if apply_nonlin is not None:
            self.apply_nonlin = eval(apply_nonlin)
        else:
            self.apply_nonlin = None
        self.smooth = smooth

    def forward(self, x=None, y=None, loss_mask=None, logits=None, target=None, **kwargs):
        if x is None: x = logits
        if y is None: y = target
        
        x = einops.rearrange(x, "b h w c -> b c h w")
        y = einops.rearrange(y, "b h w c -> b c h w")
        # x: BCHW, y: BCHW
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]

        with torch.no_grad():
            loss_ret = {}
            if self.batch_dice:
                for _class in range(dc.shape[0]):
                    loss_ret[f"mask_{_class}"] = 1 - dc[_class].item()
            else:
                for _class in range(dc.shape[1]):
                    loss_ret[f"mask_{_class}"] = 1 - dc[:, _class].mean().item()
            
            # Log the overall dice loss (Mean of per-class dice losses)
            loss_ret["overall"] = 1 - dc.mean().item()

        return -dc, loss_ret


class GeneralizedDiceLoss(torch.nn.Module):
    """
    Generalized Dice Loss with inverse-squared volume weights.
    """
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., weight_eps=1e-6):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        if apply_nonlin is not None:
            self.apply_nonlin = eval(apply_nonlin)
        else:
            self.apply_nonlin = None
        self.smooth = smooth
        self.weight_eps = weight_eps

    def _to_bchw(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4:
            # Heuristic: if last dim looks like channels, assume BHWC
            if tensor.shape[-1] < tensor.shape[1]:
                return einops.rearrange(tensor, "b h w c -> b c h w")
        elif tensor.dim() == 5:
            # Heuristic: if last dim looks like channels, assume BHWDC
            if tensor.shape[-1] < tensor.shape[1]:
                return einops.rearrange(tensor, "b h w d c -> b c h w d")
        return tensor

    def forward(self, x=None, y=None, loss_mask=None, logits=None, target=None, logits_cf=None, target_cf=None, **kwargs):
        if x is None:
            x = logits_cf if logits_cf is not None else logits
        if y is None:
            y = target_cf if target_cf is not None else target
        if x is None or y is None:
            raise ValueError("GeneralizedDiceLoss requires logits/x and target/y.")

        x = self._to_bchw(x)

        # Convert y if it's likely one-hot BHWC/BHWDC; otherwise let get_tp_fp_fn_tn handle labels
        num_classes = x.shape[1]
        if y.dim() == x.dim():
            if y.dim() == 4 and y.shape[-1] == num_classes and y.shape[-1] < y.shape[1]:
                y = einops.rearrange(y, "b h w c -> b c h w")
            elif y.dim() == 5 and y.shape[-1] == num_classes and y.shape[-1] < y.shape[1]:
                y = einops.rearrange(y, "b h w d c -> b c h w d")

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        shp_x = x.shape
        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        # Class volumes from gt: sum_i g_ci = tp + fn
        g_sum = tp + fn
        weights = torch.where(
            g_sum > 0,
            1.0 / (g_sum * g_sum + self.weight_eps),
            torch.zeros_like(g_sum),
        )

        if not self.do_bg:
            if self.batch_dice:
                tp, fp, fn, weights = tp[1:], fp[1:], fn[1:], weights[1:]
            else:
                tp, fp, fn, weights = tp[:, 1:], fp[:, 1:], fn[:, 1:], weights[:, 1:]

        numerator = 2.0 * (weights * tp).sum(dim=0 if self.batch_dice else 1) + self.smooth
        denominator = (weights * (2.0 * tp + fp + fn)).sum(dim=0 if self.batch_dice else 1) + self.smooth
        gdl = numerator / (denominator + 1e-8)
        loss = 1.0 - gdl

        with torch.no_grad():
            loss_ret = {"overall": loss.mean().item()}

        return loss, loss_ret


class BoundaryLoss(torch.nn.Module):
    def __init__(self,
                 idc=None,
                 do_bg: bool = False,
                 channel_last: bool = True,
                 dist_map_power: float = 1.0,
                 dist_clip: float = None,
                 use_precomputed: bool = True,
                 use_probabilities: bool = False,
                 compute_missing_dist: bool = True,
                 warn_if_missing_precomputed: bool = True):
        super().__init__()
        self.idc = idc
        self.do_bg = do_bg
        self.channel_last = channel_last
        self.dist_map_power = dist_map_power
        self.dist_clip = dist_clip
        self.use_precomputed = use_precomputed
        self.use_probabilities = use_probabilities
        self.compute_missing_dist = compute_missing_dist
        self.warn_if_missing_precomputed = warn_if_missing_precomputed
        self._warned_missing_precomputed = False

    def _to_bchw(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 4:
            return tensor
        if self.channel_last:
            return tensor.permute(0, 3, 1, 2)
        return tensor

    def _select_classes(self, num_classes: int):
        if self.idc is not None:
            return [int(c) for c in self.idc]
        if not self.do_bg:
            return list(range(1, num_classes))
        return None

    def forward(self, logits=None, target=None, probabilities=None, target_dist=None,
                logits_cf=None, target_cf=None, probabilities_cf=None, target_dist_cf=None, **kwargs):
        if self.use_probabilities and probabilities is not None:
            probs = probabilities_cf if probabilities_cf is not None else probabilities
            probs = self._to_bchw(probs)
        elif probabilities is None or not self.use_probabilities:
            if logits_cf is not None:
                probs = F.softmax(logits_cf, dim=1)
            else:
                if logits is None:
                    raise ValueError("BoundaryLoss requires logits or probabilities.")
                probs = F.softmax(logits, dim=-1 if self.channel_last else 1)
                probs = self._to_bchw(probs)

        num_classes = probs.shape[1]

        if target is None:
            raise ValueError("BoundaryLoss requires target.")

        if target_cf is not None:
            target = target_cf.float()
        elif target.dim() == 3:
            target = F.one_hot(target.long(), num_classes=num_classes).float()
            target = target.permute(0, 3, 1, 2)
        else:
            target = self._to_bchw(target).float()

        class_ids = self._select_classes(num_classes)
        if class_ids is not None and len(class_ids) == 0:
            zero = torch.zeros(1, device=probs.device, dtype=probs.dtype).mean()
            return zero, {"loss": 0.0}
        if class_ids is not None:
            probs = probs[:, class_ids]

        dist = None
        if self.use_precomputed and (target_dist_cf is not None or target_dist is not None):
            dist = target_dist_cf if target_dist_cf is not None else target_dist
            if dist.dim() == 4 and dist.shape[-1] < dist.shape[1]:
                dist = dist.permute(0, 3, 1, 2)
            if class_ids is not None and dist.shape[1] != len(class_ids):
                dist = dist[:, class_ids]
            dist = dist.to(device=probs.device, dtype=probs.dtype)
            if dist.shape[-2:] != probs.shape[-2:]:
                dist = None

        if dist is None:
            if self.use_precomputed and not self.compute_missing_dist:
                if self.warn_if_missing_precomputed and not self._warned_missing_precomputed:
                    print(
                        "[BoundaryLoss] target_dist missing while use_precomputed=True. "
                        "Skipping BoundaryLoss for this batch. "
                        "Provide `target_dist` (preferred) or set compute_missing_dist=True."
                    )
                    self._warned_missing_precomputed = True
                zero = torch.zeros(1, device=probs.device, dtype=probs.dtype).mean()
                return zero, {"loss": 0.0}
            with torch.no_grad():
                target_np = target.detach().cpu().numpy()
                dist_np = batch_signed_distance_map(
                    target_np,
                    class_ids=class_ids,
                    channel_first=True,
                    compact=True,
                    dtype=np.float32,
                )
                dist = torch.from_numpy(dist_np).to(device=probs.device, dtype=probs.dtype)

        if self.dist_map_power != 1.0:
            dist = dist.sign() * dist.abs().pow(self.dist_map_power)
        if self.dist_clip is not None:
            dist = dist.clamp(-self.dist_clip, self.dist_clip)

        loss = torch.mean(probs * dist)
        return loss, {"loss": loss.item()}
