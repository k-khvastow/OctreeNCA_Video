"""
Agent for the flow-augmented dual-view warm-start OctreeNCA.

Extends OctreeNCADualViewWarmStartM1InitAgent with:
  - Photometric flow loss computed from model outputs
  - Flow magnitude logging to W&B / tensorboard
  - Warp-quality visualisation during evaluation
"""

import warnings

import numpy as np
import torch

from src.agents.Agent_OctreeNCA_DualView_WarmStart_M1Init import (
    OctreeNCADualViewWarmStartM1InitAgent,
)
from src.losses.PhotometricFlowLoss import PhotometricFlowLoss


class OctreeNCADualViewFlowAgent(OctreeNCADualViewWarmStartM1InitAgent):
    """Temporal dual-view warm-start agent with self-supervised flow loss."""

    def __init__(self, model):
        super().__init__(model)
        self._flow_loss_fn = PhotometricFlowLoss(
            photometric_weight=1.0,
            ssim_weight=0.0,
            smoothness_weight=0.01,
            edge_aware_smoothness=True,
        )

    def _compute_flow_loss(self, out: dict, config: dict):
        """Compute photometric flow loss if the model produced flow fields.

        Returns (flow_loss_tensor, flow_loss_dict) or (None, {}).
        """
        flow_a = out.get("flow_a", None)
        flow_b = out.get("flow_b", None)
        prev_img_a = out.get("prev_img_a", None)
        prev_img_b = out.get("prev_img_b", None)
        current_img_a = out.get("current_img_a", None)
        current_img_b = out.get("current_img_b", None)

        if flow_a is None or prev_img_a is None or current_img_a is None:
            return None, {}

        flow_weight = float(config.get("model.flow.loss_weight", 0.1))
        smooth_weight = float(config.get("model.flow.smoothness_weight", 0.01))
        ssim_weight = float(config.get("model.flow.ssim_weight", 0.0))

        # Update the loss function weights from config (in case changed at runtime)
        self._flow_loss_fn.photometric_weight = 1.0
        self._flow_loss_fn.smoothness_weight = smooth_weight
        self._flow_loss_fn.ssim_weight = ssim_weight

        flow_loss, flow_dict = self._flow_loss_fn(
            flow_a, flow_b,
            prev_img_a, prev_img_b,
            current_img_a, current_img_b,
        )

        # Scale total flow loss by the global flow weight
        return flow_loss * flow_weight, flow_dict

    def batch_step(self, data: dict, loss_f: torch.nn.Module) -> dict:
        """Override batch_step to inject flow loss into the training loop.

        Strategy: We intercept the model output at each temporal step,
        compute the photometric flow loss, and add it to the accumulated
        segmentation loss.  This is done by monkey-patching self.model's
        forward to capture outputs — BUT to keep things clean, we instead
        replicate the relevant parts of the parent batch_step.

        For maintainability we call super().batch_step() and then
        separately account for the flow loss that was already embedded
        in the model outputs.  However, the parent doesn't know about
        flow fields.  So we take the simpler approach: override the
        whole loop.
        """
        # The parent's batch_step is complex (TBPTT, curriculum, etc.).
        # Rather than duplicating 200 lines, we use a hook-based approach:
        # wrap the loss_f to add flow loss transparently.

        original_forward = loss_f.forward
        flow_loss_accum = {"value": 0.0, "count": 0, "dict": {}}
        config = self.exp.config

        def _augmented_loss_forward(**kwargs):
            """Wrapper that adds flow loss to the standard loss."""
            # Compute standard segmentation loss
            seg_loss, seg_dict = original_forward(**kwargs)

            # Extract flow outputs from kwargs (passed through from model output)
            flow_a = kwargs.get("flow_a", None)
            if flow_a is None:
                return seg_loss, seg_dict

            flow_loss, flow_dict = self._compute_flow_loss(kwargs, config)
            if flow_loss is not None and isinstance(flow_loss, torch.Tensor):
                combined_loss = seg_loss + flow_loss
                # Merge loss dicts
                combined_dict = dict(seg_dict)
                for k, v in flow_dict.items():
                    combined_dict[k] = v
                # Accumulate for logging
                flow_loss_accum["value"] += flow_dict.get("FlowLoss/total", 0.0)
                flow_loss_accum["count"] += 1
                for k, v in flow_dict.items():
                    if k not in flow_loss_accum["dict"]:
                        flow_loss_accum["dict"][k] = 0.0
                    flow_loss_accum["dict"][k] += v
                return combined_loss, combined_dict

            return seg_loss, seg_dict

        # Temporarily replace loss forward
        loss_f.forward = _augmented_loss_forward
        try:
            loss_ret = super().batch_step(data, loss_f)
        finally:
            loss_f.forward = original_forward

        # Average accumulated flow losses and merge into return dict
        if flow_loss_accum["count"] > 0:
            n = flow_loss_accum["count"]
            for k, v in flow_loss_accum["dict"].items():
                loss_ret[k] = v / n

        return loss_ret
