import torch
import torch.nn as nn


class ContractiveRegularization(nn.Module):
    r"""
    Contractive regularization for NCA latent-space dynamics.

    Penalises the squared Frobenius norm of the Jacobian of the NCA update
    function with respect to the hidden (latent) channels of the state.

    .. math::

        \mathcal{L}_{\text{contr}} = \frac{1}{N} \sum_i \| J_i \|_F^2

    where :math:`J_i = \partial f(h_i) / \partial h_i` is the Jacobian of the
    update for sample *i*.

    Computing the full Jacobian is prohibitively expensive for spatial states
    (it would be ``C_hidden * H * W`` backward passes).  Instead we use the
    **Hutchinson trace estimator**: draw a random probe vector
    :math:`v \sim \mathcal{N}(0, I)` (or Rademacher), compute
    :math:`J^\top v` via a single ``torch.autograd.grad`` call, and use
    :math:`\|J^\top v\|^2` as an unbiased estimate of
    :math:`\|J\|_F^2 = \operatorname{tr}(J^\top J)`.

    Multiple probes (``n_probes``) reduce variance at the cost of extra
    backward passes.

    Args:
        n_probes: Number of random probe vectors for the Hutchinson estimator
            (default: 1).  More probes → lower variance but more compute.
        probe_dist: Distribution for probe vectors: ``"normal"`` (default) or
            ``"rademacher"``.
        reduction: ``"mean"`` (default) or ``"sum"``.
    """

    def __init__(
        self,
        n_probes: int = 1,
        probe_dist: str = "normal",
        reduction: str = "mean",
        **kwargs,
    ) -> None:
        super().__init__()
        self.n_probes = max(1, n_probes)
        self.probe_dist = probe_dist.strip().lower()
        self.reduction = reduction
        if self.probe_dist not in ("normal", "rademacher"):
            raise ValueError(
                f"probe_dist must be 'normal' or 'rademacher', got '{self.probe_dist}'."
            )

    @staticmethod
    def _sample_probe(like: torch.Tensor, dist: str) -> torch.Tensor:
        """Sample a random probe vector with the same shape as ``like``."""
        if dist == "rademacher":
            return torch.empty_like(like).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        return torch.randn_like(like)

    def forward(
        self,
        output_hidden: torch.Tensor,
        input_hidden: torch.Tensor,
        **kwargs,
    ) -> tuple:
        """
        Compute the contractive regularization loss.

        Both tensors must be part of the **same computation graph** (i.e.
        ``input_hidden`` was used to compute ``output_hidden`` and
        ``input_hidden.requires_grad`` is True).

        Args:
            output_hidden: Hidden channels **after** the NCA update,
                shape ``(B, *spatial)`` — any layout is fine as long as both
                tensors share the same shape.
            input_hidden:  Hidden channels **before** the NCA update (must
                require grad and be a leaf or intermediate of the graph that
                produced ``output_hidden``).

        Returns:
            ``(loss_scalar, {"contractive_frob_sq": float_value})``
        """
        if not input_hidden.requires_grad:
            # Gradient computation needs grad on the input.  If the tensor was
            # detached (e.g. between TBPTT chunks) we cannot compute J; return
            # zero so training continues unaffected.
            zero = output_hidden.new_tensor(0.0)
            return zero, {"contractive_frob_sq": 0.0}

        frob_sq_sum = output_hidden.new_tensor(0.0)
        for _ in range(self.n_probes):
            v = self._sample_probe(output_hidden, self.probe_dist)
            # J^T v  via vector-Jacobian product (backward-mode AD)
            (Jt_v,) = torch.autograd.grad(
                outputs=output_hidden,
                inputs=input_hidden,
                grad_outputs=v,
                create_graph=True,   # need second-order grad for backprop
                retain_graph=True,
                allow_unused=False,
            )
            # ||J^T v||^2  is an unbiased estimate of ||J||_F^2
            frob_sq_sum = frob_sq_sum + (Jt_v * Jt_v).sum()

        # Average over probes and batch
        batch_size = output_hidden.shape[0]
        n_elements = output_hidden[0].numel()  # spatial * channels per sample

        if self.reduction == "mean":
            loss = frob_sq_sum / (self.n_probes * batch_size * n_elements)
        elif self.reduction == "sum":
            loss = frob_sq_sum / self.n_probes
        else:
            raise ValueError(f"Unknown reduction '{self.reduction}'.")

        return loss, {"contractive_frob_sq": loss.item()}
