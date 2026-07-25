import warnings
import math
from typing import Iterable

import torch
import torch.optim
import numpy as np


class LMOptimizer(torch.optim.Optimizer):
    """Levenberg-Marquardt optimizer to be used with custom closure objects."""

    def __init__(
            self,
            params: Iterable,
            lm_parameter: float = 1.0,
            damped_term_min: float = 1e-4,
            tolerance: float = 2.0,
            lam_increase_factor: float = 2.0,
            lam_decrease_factor: float = 2.0,
            lam_eps: float = 1e-6,
            beta: float = 0.99,
    ):
        """Constructor.

        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lm_parameter: Initial value of the Levenberg-Marquardt parameter.
            damped_term_min: Minimum value of the damped term (to avoid large jumps when gradients are small).
            tolerance: If the scalar loss increases by more than this factor, the step is rejected.
            lam_increase_factor: Factor by which to increase the LM parameter when the loss increases.
            lam_decrease_factor: Factor by which to decrease the LM parameter when the loss decreases.
            lam_eps: LM parameter is bounded within [lam_eps, lam_eps ** -1].
            beta: Multiplier for running mean of damping matrix.
        """
        defaults = {
            "lm_parameter": lm_parameter,
            "damped_term_min": damped_term_min,
            "tolerance": tolerance,
            "lam_increase_factor": lam_increase_factor,
            "lam_decrease_factor": lam_decrease_factor,
            "lam_eps": lam_eps,
            "beta": beta,
            "damping_terms": None,
        }
        super(LMOptimizer, self).__init__(params, defaults)
        self.logs = {}

    @torch.no_grad()
    def step(self, closure):
        """Perform a single LM optimization step, evaluate the loss and conditionally update the parameters.

        Args:
            closure (callable): An object with a function to return the necessary data for LM optimization.
        """
        closure = closure.args[-1]  # Remove wrapper
        logs = {}

        assert len(self.param_groups) == 1, (
            "LMOptimizer does not support per-parameter options (parameter groups)."
        )
        group = self.param_groups[0]

        lam = group["lm_parameter"]
        min_damp = group["damped_term_min"]
        logs["lm_parameter"] = lam

        with torch.enable_grad():
            loss, scalar_loss, jacobian, constraint_mask = (
                closure.get_least_squares_quantities()
            )

        residual_jacobian = jacobian[~constraint_mask]
        constraint_jacobian = jacobian[constraint_mask]
        residuals = loss[~constraint_mask]
        constraints = loss[constraint_mask]
        n_variables = jacobian.shape[1]
        logs["n_residuals"] = float(residuals.numel())

        # Compute damping terms
        damping_terms = residual_jacobian.norm(
            dim=0
        )  # Compute sqrt of diagonal elements of pseudo-Hessian
        if group["damping_terms"] is None:
            damping_terms = damping_terms.clamp(min=min_damp)
        else:
            beta = group["beta"]
            damping_terms = (
                    beta * torch.max(damping_terms, group["damping_terms"])
                    + (1 - beta) * damping_terms
            )

        if (
                constraints.numel() > 0
        ):  # If there are constraints, solve constrained optimization problem
            # Compute pseudo-Hessian and add damped term to it
            pseudo_hessian = residual_jacobian.T @ residual_jacobian
            damping_matrix = (damping_terms ** 2).diag_embed()
            mat = pseudo_hessian + lam * damping_matrix
            b = -residual_jacobian.T @ residuals

            # Update mat and b with Lagrange multipliers for constraints
            mat = torch.nn.functional.pad(
                mat, (0, constraint_jacobian.shape[0], 0, constraint_jacobian.shape[0])
            )
            mat[n_variables:, :n_variables] = constraint_jacobian
            mat[:n_variables, n_variables:] = constraint_jacobian.T
            b = torch.cat((b, -constraints), dim=0)

            # Compute step
            lstsq = torch.linalg.lstsq(mat, b)
            step = lstsq[0]
            step = step[:n_variables]
        else:  # If there are no constraints, solve unconstrained optimization problem
            # Pseudo-Hessian is not computed explicitly; we solve the least-squares problem directly
            matrix = torch.cat(
                (residual_jacobian, np.sqrt(lam) * damping_terms.diag_embed()), dim=0
            )
            bb = torch.cat((residuals, torch.zeros_like(damping_terms)), dim=0)
            # Least-squares solver on CPU is more reliable
            try:
                lstsq = torch.linalg.lstsq(matrix.cpu(), -bb.cpu(), driver="gelsd")
                step = lstsq[0].to(matrix.device)
                logs["rank"] = float(lstsq.rank.item())
            except RuntimeError:
                warnings.warn("Least-squares solver failed; step ignored")
                step = None

        # Update parameters
        if step is not None:
            trainable_params = [v for v in group["params"] if v.requires_grad]
            assert len(trainable_params) == 1, (
                "LMOptimizer does not support multiple parameter tensors"
            )
            p = trainable_params[0]
            p_copy = p.data.clone()
            p.data.add_(step)

            with torch.inference_mode(True):
                updated_loss = closure.evaluate_least_squares_loss()
            loss_ratio = (updated_loss / scalar_loss).item()
            logs["loss_ratio"] = loss_ratio
            if not math.isfinite(loss_ratio) or loss_ratio > group["tolerance"]:
                # Reject step
                p.data = p_copy
            if loss_ratio <= 1.0:
                lam = lam / group["lam_decrease_factor"]
            else:
                lam = lam * group["lam_increase_factor"]

            # Update optimizer parameters
            group["lm_parameter"] = np.clip(lam, group["lam_eps"], 1 / group["lam_eps"])
            group["damping_terms"] = damping_terms

        # Log
        self.logs = logs

        return scalar_loss
