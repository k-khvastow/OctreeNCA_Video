"""
M1+M2 wrapper where M2 is the flow-augmented warm-start model.

Identical to OctreeNCA2DDualViewWarmStartM1Init except M2 is an instance
of OctreeNCA2DDualViewWarmStartFlow (with optical-flow warping).
"""

from typing import Optional

import torch

from src.models.Model_OctreeNCA_2d_dual_view_warm_m1init import OctreeNCA2DDualViewWarmStartM1Init
from src.models.Model_OctreeNCA_2d_dual_view_warm_flow import OctreeNCA2DDualViewWarmStartFlow


class OctreeNCA2DDualViewWarmStartFlowM1Init(OctreeNCA2DDualViewWarmStartM1Init):
    """Two-model temporal dual-view setup with flow-augmented M2.

    - M1 (cold model) initialises recurrent states at t=0.
    - M2 (warm + flow model) warps previous state, refines with NCA, and
      outputs a self-supervised flow loss signal alongside segmentation.
    """

    def __init__(self, config: dict):
        # We need to override _build_m2 before super().__init__ is done.
        # The simplest way: call grandparent init, then build M1/M2 ourselves.
        # But the parent __init__ does a lot of wiring.  Instead, we call
        # super().__init__ which builds M2 as OctreeNCA2DDualViewWarmStart,
        # then we REPLACE self.m2 with the flow variant.
        super().__init__(config)

        # Rebuild M2 as the flow-augmented variant
        m2_num_levels = config.get("model.m2.num_levels", None)
        if m2_num_levels is not None:
            m2_config = dict(config)
            full_res = config["model.octree.res_and_steps"]
            m2_levels = int(m2_num_levels)
            m2_config["model.octree.res_and_steps"] = full_res[:m2_levels]
            ks = m2_config.get("model.kernel_size", None)
            if isinstance(ks, (list, tuple)) and len(ks) > m2_levels:
                m2_config["model.kernel_size"] = list(ks)[:m2_levels]
            ps = m2_config.get("model.train.patch_sizes", None)
            if isinstance(ps, (list, tuple)) and len(ps) > m2_levels:
                m2_config["model.train.patch_sizes"] = list(ps)[:m2_levels]
        else:
            m2_config = config

        self.m2 = OctreeNCA2DDualViewWarmStartFlow(m2_config)

        # Re-sync attributes
        self.channel_n = self.m2.channel_n
        self.input_channels = self.m2.input_channels
        self.output_channels = self.m2.output_channels

        # Re-apply M2 init from M1 / identity init / shared backbone
        self._sync_m2_from_m1()
