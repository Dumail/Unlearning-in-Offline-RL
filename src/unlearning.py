"""Gradient Ascent + Head Refit unlearning for Decision Transformer.

Phase 1 (Gradient Ascent): Maximize NLL on D_f while staying close to base model
  via forward KL penalty on D_r. Only body parameters are updated; variance and
  action head are frozen.

Phase 2 (Head Refit): Re-initialize the action head and retrain it on D_r with
  the body frozen. This allows the head to re-learn clean action predictions
  from the modified body representations.

"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import TrajectoryDataset
from .decision_transformer import DecisionTransformer
from .nll import compute_trajectory_nll, gaussian_nll


def _get_body_param_names(model: DecisionTransformer) -> set[str]:
    """Get names of body parameters (everything except action head and variance)."""
    excluded = {
        "predict_action_mean.weight",
        "predict_action_mean.bias",
        "action_log_var",
    }
    return {n for n, _ in model.named_parameters() if n not in excluded}


class GradientAscentUnlearner:
    """Gradient Ascent + KL regularization + Head Refit."""

    def __init__(
        self,
        model: DecisionTransformer,
        base_model: DecisionTransformer,
        forget_dataset: TrajectoryDataset,
        retain_dataset: TrajectoryDataset,
        kl_weight: float = 1.0,
        lr: float = 1e-4,
        grad_clip: float = 0.25,
        batch_size: int = 64,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.kl_weight = kl_weight
        self.grad_clip = grad_clip

        # Base model is frozen (reference for KL penalty)
        self.base_model = base_model.to(device)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Freeze variance (prevent trivial variance inflation)
        self.model.action_log_var.requires_grad = False

        # Freeze action head during ascent (body-only update)
        for p in self.model.predict_action_mean.parameters():
            p.requires_grad = False

        # Optimizer for body parameters only
        body_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(body_params, lr=lr)

        # Data loaders
        self.forget_loader = DataLoader(
            forget_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self.retain_loader = DataLoader(
            retain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self._forget_iter = iter(self.forget_loader)
        self._retain_iter = iter(self.retain_loader)

    def _get_batch(self, iterator, loader) -> dict:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
            # Update stored iterator
            if loader is self.forget_loader:
                self._forget_iter = iterator
            else:
                self._retain_iter = iterator
        return {k: v.to(self.device) for k, v in batch.items()}

    def ascent_step(self) -> dict:
        """One step of gradient ascent on D_f with KL penalty on D_r.

        Loss = -NLL(D_f) + lambda * KL(pi_base || pi_unlearn) on D_r
        Minimizing this = ascending on NLL + descending on KL.
        """
        self.model.train()

        # --- Forget batch: maximize NLL ---
        f_batch = self._get_batch(self._forget_iter, self.forget_loader)
        f_mean, _ = self.model(
            f_batch["states"],
            f_batch["actions"],
            f_batch["returns_to_go"],
            f_batch["timesteps"],
            f_batch["attention_mask"],
        )
        forget_nll = gaussian_nll(
            f_mean,
            f_batch["actions"],
            self.model.action_log_var,
            f_batch["attention_mask"],
        )

        # --- Retain batch: minimize KL(base || unlearn) ---
        r_batch = self._get_batch(self._retain_iter, self.retain_loader)
        r_mean_unlearn, _ = self.model(
            r_batch["states"],
            r_batch["actions"],
            r_batch["returns_to_go"],
            r_batch["timesteps"],
            r_batch["attention_mask"],
        )
        with torch.no_grad():
            r_mean_base, _ = self.base_model(
                r_batch["states"],
                r_batch["actions"],
                r_batch["returns_to_go"],
                r_batch["timesteps"],
                r_batch["attention_mask"],
            )

        # KL for Gaussians with same variance: sum_d (mu_base - mu_unlearn)^2 / (2 * sigma_d^2)
        sigma_sq = torch.clamp(torch.exp(self.model.action_log_var), min=1e-4)
        kl_per_token = 0.5 * ((r_mean_base - r_mean_unlearn) ** 2 / sigma_sq).sum(
            dim=-1
        )
        mask = r_batch["attention_mask"]
        kl = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)

        # Combined loss
        loss = -forget_nll + self.kl_weight * kl

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.grad_clip,
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "forget_nll": forget_nll.item(),
            "kl": kl.item(),
        }

    def run_ascent(self, n_steps: int) -> list[dict]:
        """Run N steps of gradient ascent. Returns list of step metrics."""
        metrics_log = []
        log_interval = max(1, n_steps // 10)

        for step in range(n_steps):
            metrics = self.ascent_step()
            metrics_log.append(metrics)

            if step % log_interval == 0 or step == n_steps - 1:
                pct = step / n_steps * 100
                print(
                    f"[Ascent {pct:5.1f}%] step {step}/{n_steps}  "
                    f"loss={metrics['loss']:.4f}  "
                    f"nll={metrics['forget_nll']:.4f}  "
                    f"kl={metrics['kl']:.6f}"
                )

        return metrics_log

    def refit_head(
        self,
        retain_dataset: TrajectoryDataset,
        n_steps: int = 10000,
        lr: float = 1e-4,
        batch_size: int = 64,
        reinit_head: bool = True,
    ) -> list[dict]:
        """Re-initialize action head and refit on D_r with frozen body.

        Returns list of step metrics.
        """
        print("\n--- Head Refit Phase ---")

        if reinit_head:
            self.model.predict_action_mean.apply(self.model._init_weights)

        # Freeze everything, unfreeze only action head
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.predict_action_mean.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(
            self.model.predict_action_mean.parameters(),
            lr=lr,
        )
        loader = DataLoader(
            retain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        loader_iter = iter(loader)

        metrics_log = []
        log_interval = max(1, n_steps // 10)

        # LR schedule: warmup then cosine
        warmup = min(1000, n_steps // 10)

        for step in range(n_steps):
            # LR schedule
            if step < warmup:
                cur_lr = lr * step / max(1, warmup)
            else:
                progress = (step - warmup) / max(1, n_steps - warmup)
                cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            action_mean, _ = self.model(
                batch["states"],
                batch["actions"],
                batch["returns_to_go"],
                batch["timesteps"],
                batch["attention_mask"],
            )
            loss = gaussian_nll(
                action_mean,
                batch["actions"],
                self.model.action_log_var,
                batch["attention_mask"],
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.predict_action_mean.parameters(),
                0.25,
            )
            optimizer.step()

            metrics = {"refit_loss": loss.item(), "refit_lr": cur_lr}
            metrics_log.append(metrics)

            if step % log_interval == 0 or step == n_steps - 1:
                pct = step / n_steps * 100
                print(
                    f"[Refit {pct:5.1f}%] step {step}/{n_steps}  "
                    f"loss={loss.item():.4f}  lr={cur_lr:.6f}"
                )

        # Restore requires_grad for all parameters
        for p in self.model.parameters():
            p.requires_grad = True
        # Keep variance frozen flag consistent
        self.model.action_log_var.requires_grad = False

        return metrics_log


def build_reward_flipped_trajectories(
    trajectories: list[dict],
) -> list[dict]:
    flipped_trajectories = []
    for traj in trajectories:
        flipped_rewards = -np.array(traj["rewards"], copy=True)
        flipped_traj = dict(traj)
        flipped_traj["rewards"] = flipped_rewards
        if "return" in flipped_traj:
            flipped_traj["return"] = float(flipped_rewards.sum())
        flipped_trajectories.append(flipped_traj)
    return flipped_trajectories


def build_reward_flipped_dataset(
    trajectories: list[dict],
    context_length: int,
    state_mean: np.ndarray | None,
    state_std: np.ndarray | None,
    antmaze_goal_mode: str = "none",
    antmaze_offline_state_mode: str = "observations",
    antmaze_reward_mode: str = "none",
) -> TrajectoryDataset:
    return TrajectoryDataset(
        build_reward_flipped_trajectories(trajectories),
        context_length=context_length,
        state_mean=state_mean,
        state_std=state_std,
        antmaze_goal_mode=antmaze_goal_mode,
        antmaze_offline_state_mode=antmaze_offline_state_mode,
        antmaze_reward_mode=antmaze_reward_mode,
    )


def _gaussian_kl_same_variance(
    teacher_mean: torch.Tensor,
    student_mean: torch.Tensor,
    log_var: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    sigma_sq = torch.clamp(torch.exp(log_var), min=1e-4)
    kl_per_token = 0.5 * ((teacher_mean - student_mean) ** 2 / sigma_sq).sum(dim=-1)
    return (kl_per_token * attention_mask).sum() / attention_mask.sum().clamp(min=1)


class TrajDeleterUnlearner:
    def __init__(
        self,
        model: DecisionTransformer,
        base_model: DecisionTransformer,
        forget_dataset_flipped: TrajectoryDataset,
        retain_dataset: TrajectoryDataset,
        alpha: float = 1.0,
        beta: float = 1.0,
        stage1_lr: float = 1e-4,
        grad_clip: float = 0.25,
        batch_size: int = 64,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self._retain_dataset = retain_dataset

        self.base_model = base_model.to(device)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.model.action_log_var.requires_grad = False

        stage1_params = [p for p in self.model.parameters() if p.requires_grad]
        self.stage1_optimizer = torch.optim.Adam(stage1_params, lr=stage1_lr)

        self.forget_loader = DataLoader(
            forget_dataset_flipped,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self.retain_loader = DataLoader(
            retain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self._forget_iter = iter(self.forget_loader)
        self._retain_iter = iter(self.retain_loader)

    def _get_batch(self, iterator, loader) -> dict:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
            if loader is self.forget_loader:
                self._forget_iter = iterator
            else:
                self._retain_iter = iterator
        return {k: v.to(self.device) for k, v in batch.items()}

    def stage1_step(self) -> dict:
        self.model.train()

        f_batch = self._get_batch(self._forget_iter, self.forget_loader)
        f_mean, _ = self.model(
            f_batch["states"],
            f_batch["actions"],
            f_batch["returns_to_go"],
            f_batch["timesteps"],
            f_batch["attention_mask"],
        )
        forget_flipped_nll = gaussian_nll(
            f_mean,
            f_batch["actions"],
            self.model.action_log_var,
            f_batch["attention_mask"],
        )

        r_batch = self._get_batch(self._retain_iter, self.retain_loader)
        r_mean, _ = self.model(
            r_batch["states"],
            r_batch["actions"],
            r_batch["returns_to_go"],
            r_batch["timesteps"],
            r_batch["attention_mask"],
        )
        retain_nll = gaussian_nll(
            r_mean,
            r_batch["actions"],
            self.model.action_log_var,
            r_batch["attention_mask"],
        )

        loss = retain_nll + self.alpha * forget_flipped_nll

        self.stage1_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.grad_clip,
        )
        self.stage1_optimizer.step()

        return {
            "loss": loss.item(),
            "retain_nll": retain_nll.item(),
            "forget_flipped_nll": forget_flipped_nll.item(),
        }

    def run_stage1(self, n_steps: int) -> list[dict]:
        metrics_log = []
        log_interval = max(1, n_steps // 10)

        for step in range(n_steps):
            metrics = self.stage1_step()
            metrics_log.append(metrics)

            if step % log_interval == 0 or step == n_steps - 1:
                pct = step / n_steps * 100
                print(
                    f"[TrajDeleter-S1 {pct:5.1f}%] step {step}/{n_steps}  "
                    f"loss={metrics['loss']:.4f}  "
                    f"retain_nll={metrics['retain_nll']:.4f}  "
                    f"forget_flipped_nll={metrics['forget_flipped_nll']:.4f}"
                )

        return metrics_log

    def run_stage2(
        self,
        n_steps: int,
        lr: float = 1e-4,
        batch_size: int | None = None,
    ) -> list[dict]:
        print("\n--- TrajDeleter Stage 2 ---")

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.action_log_var.requires_grad = False

        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=lr,
        )
        loader = DataLoader(
            self._retain_dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        loader_iter = iter(loader)

        metrics_log = []
        log_interval = max(1, n_steps // 10)
        warmup = min(1000, n_steps // 10)

        for step in range(n_steps):
            if step < warmup:
                cur_lr = lr * step / max(1, warmup)
            else:
                progress = (step - warmup) / max(1, n_steps - warmup)
                cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            student_mean, _ = self.model(
                batch["states"],
                batch["actions"],
                batch["returns_to_go"],
                batch["timesteps"],
                batch["attention_mask"],
            )
            retain_nll = gaussian_nll(
                student_mean,
                batch["actions"],
                self.model.action_log_var,
                batch["attention_mask"],
            )

            with torch.no_grad():
                teacher_mean, _ = self.base_model(
                    batch["states"],
                    batch["actions"],
                    batch["returns_to_go"],
                    batch["timesteps"],
                    batch["attention_mask"],
                )

            anchor_kl = _gaussian_kl_same_variance(
                teacher_mean,
                student_mean,
                self.model.action_log_var,
                batch["attention_mask"],
            )
            loss = retain_nll + self.beta * anchor_kl

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.grad_clip,
            )
            optimizer.step()

            metrics = {
                "loss": loss.item(),
                "retain_nll": retain_nll.item(),
                "anchor_kl": anchor_kl.item(),
                "stage2_lr": cur_lr,
            }
            metrics_log.append(metrics)

            if step % log_interval == 0 or step == n_steps - 1:
                pct = step / n_steps * 100
                print(
                    f"[TrajDeleter-S2 {pct:5.1f}%] step {step}/{n_steps}  "
                    f"loss={loss.item():.4f}  "
                    f"retain_nll={retain_nll.item():.4f}  "
                    f"anchor_kl={anchor_kl.item():.6f}  "
                    f"lr={cur_lr:.6f}"
                )

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.action_log_var.requires_grad = False

        return metrics_log


def compute_per_trajectory_difficulty(
    model: DecisionTransformer,
    forget_trajs: list[dict],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    context_length: int = 20,
    device: str = "cuda",
    antmaze_goal_mode: str = "none",
    antmaze_offline_state_mode: str = "observations",
    antmaze_reward_mode: str = "none",
) -> np.ndarray:
    """Compute difficulty score for each forget trajectory.

    Uses per-trajectory mean NLL from the base model as a membership strength proxy.
    Lower NLL -> model is more confident about the trajectory -> stronger memorization -> harder to forget.
    Returns negated values (-NLL), so higher values = harder to forget.

    Returns: shape (n_trajectories,), higher values indicate harder-to-forget trajectories.
    """
    model.eval()
    nlls = []
    for traj in forget_trajs:
        nll = compute_trajectory_nll(
            model,
            traj,
            state_mean,
            state_std,
            context_length=context_length,
            device=device,
            antmaze_goal_mode=antmaze_goal_mode,
            antmaze_offline_state_mode=antmaze_offline_state_mode,
            antmaze_reward_mode=antmaze_reward_mode,
        )
        nlls.append(nll)
    nlls_arr = np.array(nlls)
    # -NLL: higher = lower NLL = more confident = harder to forget
    return -nlls_arr


def _assign_group_steps(
    difficulty_scores: np.ndarray,
    total_steps: int,
    n_groups: int = 4,
) -> list[tuple[np.ndarray, int]]:
    """Assign GA steps by difficulty group.

    Harder groups receive more steps, easier groups receive fewer.
    Allocation weights are proportional to softmax of mean difficulty within each group.

    Returns: [(group_indices, allocated_steps), ...] sorted from hardest to easiest.
    """
    n = len(difficulty_scores)
    # Sort by difficulty descending
    sorted_indices = np.argsort(-difficulty_scores)
    group_size = max(1, n // n_groups)

    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else n
        if start >= n:
            break
        indices = sorted_indices[start:end]
        mean_diff = difficulty_scores[indices].mean()
        groups.append((indices, mean_diff))

    if not groups:
        return []

    # Use softmax to compute weights, temperature=1
    mean_diffs = np.array([g[1] for g in groups], dtype=np.float64)
    # Normalize to [0, 1] to avoid numerical issues
    if mean_diffs.max() - mean_diffs.min() > 1e-8:
        normalized = (mean_diffs - mean_diffs.min()) / (
            mean_diffs.max() - mean_diffs.min()
        )
    else:
        normalized = np.ones_like(mean_diffs)
    # Exponential weighting: harder groups get more steps
    weights = np.exp(normalized)
    weights = weights / weights.sum()

    result = []
    remaining = total_steps
    for i, (indices, _) in enumerate(groups):
        if i == len(groups) - 1:
            steps = remaining
        else:
            steps = max(1, int(total_steps * weights[i]))
            remaining -= steps
        result.append((indices, max(1, steps)))

    return result


class AdaptiveBudgetUnlearner:
    """Subgroup-Adaptive Budget Allocation unlearning.

    Core idea: partition the forget set by per-trajectory difficulty into groups,
    assign unequal GA steps to different difficulty subsets (harder ones get more),
    then perform a unified head refit.
    """

    def __init__(
        self,
        model: DecisionTransformer,
        base_model: DecisionTransformer,
        forget_trajs: list[dict],
        retain_dataset: TrajectoryDataset,
        difficulty_scores: np.ndarray,
        kl_weight: float = 1.0,
        lr: float = 1e-4,
        grad_clip: float = 0.25,
        batch_size: int = 64,
        device: str = "cuda",
        n_groups: int = 4,
        context_length: int = 20,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
    ):
        self.model = model
        self.base_model = base_model
        self.forget_trajs = forget_trajs
        self.retain_dataset = retain_dataset
        self.difficulty_scores = difficulty_scores
        self.kl_weight = kl_weight
        self.lr = lr
        self.grad_clip = grad_clip
        self.batch_size = batch_size
        self.device = device
        self.n_groups = n_groups
        self.context_length = context_length
        self.state_mean = state_mean
        self.state_std = state_std

    def run_adaptive_ascent(
        self,
        total_steps: int,
    ) -> dict:
        """Execute adaptive GA grouped by difficulty.

        Returns: result dictionary containing per-group metrics.
        """
        group_plan = _assign_group_steps(
            self.difficulty_scores, total_steps, self.n_groups
        )

        print(f"\n=== Adaptive Budget Allocation ===")
        print(f"Total steps: {total_steps}, {len(group_plan)} groups")
        for i, (indices, steps) in enumerate(group_plan):
            mean_diff = self.difficulty_scores[indices].mean()
            print(
                f"  Group {i}: {len(indices)} trajectories, "
                f"difficulty={mean_diff:.4f}, allocated {steps} GA steps"
            )

        all_group_metrics = []
        total_actual_steps = 0

        for group_idx, (traj_indices, group_steps) in enumerate(group_plan):
            group_trajs = [self.forget_trajs[i] for i in traj_indices]
            mean_diff = self.difficulty_scores[traj_indices].mean()
            print(
                f"\n--- Group {group_idx}: {len(group_trajs)} trajectories, "
                f"difficulty={mean_diff:.4f}, GA {group_steps} steps ---"
            )

            # Create forget dataset for this group
            group_dataset = TrajectoryDataset(
                group_trajs,
                context_length=self.context_length,
                state_mean=self.state_mean,
                state_std=self.state_std,
            )

            # Create temporary unlearner (reuses current model state)
            unlearner = GradientAscentUnlearner(
                model=self.model,
                base_model=self.base_model,
                forget_dataset=group_dataset,
                retain_dataset=self.retain_dataset,
                kl_weight=self.kl_weight,
                lr=self.lr,
                grad_clip=self.grad_clip,
                batch_size=self.batch_size,
                device=self.device,
            )

            ascent_log = unlearner.run_ascent(group_steps)
            total_actual_steps += group_steps

            group_metrics = {
                "group_idx": group_idx,
                "n_trajectories": len(group_trajs),
                "mean_difficulty": float(mean_diff),
                "allocated_steps": group_steps,
                "final_forget_nll": ascent_log[-1]["forget_nll"]
                if ascent_log
                else float("nan"),
                "final_kl": ascent_log[-1]["kl"] if ascent_log else float("nan"),
                "traj_indices": traj_indices.tolist(),
            }
            all_group_metrics.append(group_metrics)

        return {
            "method": "adaptive_budget",
            "n_groups": len(group_plan),
            "total_steps": total_actual_steps,
            "group_metrics": all_group_metrics,
        }

    def refit_head(
        self,
        n_steps: int = 10000,
        lr: float = 1e-4,
    ) -> list[dict]:
        """Reuse the refit logic from GradientAscentUnlearner."""
        # Create a temporary unlearner just to call refit_head
        # (needs a dummy forget dataset, but refit does not use it)
        dummy_dataset = self.retain_dataset
        unlearner = GradientAscentUnlearner(
            model=self.model,
            base_model=self.base_model,
            forget_dataset=dummy_dataset,
            retain_dataset=self.retain_dataset,
            kl_weight=self.kl_weight,
            lr=self.lr,
            grad_clip=self.grad_clip,
            batch_size=self.batch_size,
            device=self.device,
        )
        return unlearner.refit_head(
            self.retain_dataset,
            n_steps=n_steps,
            lr=lr,
        )


def _build_sample_weights(
    forget_dataset: TrajectoryDataset,
    difficulty_scores: np.ndarray,
    temperature: float = 1.0,
) -> list[float]:
    """Map per-trajectory difficulty to per-sample sampling weights.

    Propagates trajectory-level difficulty to subsequence level via dataset.index_map.
    Uses softmax(difficulty / temperature) as sampling probability.
    Lower temperature concentrates sampling on harder trajectories.
    """
    # Normalize difficulty to [0, 1]
    d = difficulty_scores.copy()
    d_range = d.max() - d.min()
    if d_range > 1e-8:
        d = (d - d.min()) / d_range
    else:
        d = np.ones_like(d)

    # Softmax with temperature
    exp_d = np.exp(d / temperature)
    traj_weights = exp_d / exp_d.sum()

    # Map to per-sample weights via index_map
    sample_weights = np.zeros(len(forget_dataset), dtype=np.float64)
    for sample_idx, (traj_idx, _) in enumerate(forget_dataset.index_map):
        sample_weights[sample_idx] = traj_weights[traj_idx]

    return torch.from_numpy(sample_weights).tolist()


class WeightedGradientAscentUnlearner:
    """Difficulty-weighted Gradient Ascent + KL + Head Refit.

    Same GA+KL+Refit flow as GradientAscentUnlearner,
    but uses WeightedRandomSampler to upsample hard trajectories by difficulty.
    Maintains a single optimizer with no momentum reset issues.
    """

    def __init__(
        self,
        model: DecisionTransformer,
        base_model: DecisionTransformer,
        forget_dataset: TrajectoryDataset,
        retain_dataset: TrajectoryDataset,
        difficulty_scores: np.ndarray,
        kl_weight: float = 1.0,
        lr: float = 1e-4,
        grad_clip: float = 0.25,
        batch_size: int = 64,
        device: str = "cuda",
        temperature: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.kl_weight = kl_weight
        self.grad_clip = grad_clip
        self.temperature = temperature

        # Base model frozen
        self.base_model = base_model.to(device)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Freeze variance + action head
        self.model.action_log_var.requires_grad = False
        for p in self.model.predict_action_mean.parameters():
            p.requires_grad = False

        # Single optimizer
        body_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(body_params, lr=lr)

        # Weighted sampler for forget set
        sample_weights = _build_sample_weights(
            forget_dataset, difficulty_scores, temperature
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(forget_dataset),
            replacement=True,
        )
        self.forget_loader = DataLoader(
            forget_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self.retain_loader = DataLoader(
            retain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self._forget_iter = iter(self.forget_loader)
        self._retain_iter = iter(self.retain_loader)

        # Save difficulty statistics
        self.difficulty_scores = difficulty_scores
        self._retain_dataset = retain_dataset

    def _get_batch(self, iterator, loader) -> dict:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
            if loader is self.forget_loader:
                self._forget_iter = iterator
            else:
                self._retain_iter = iterator
        return {k: v.to(self.device) for k, v in batch.items()}

    def ascent_step(self) -> dict:
        """Same logic as GradientAscentUnlearner.ascent_step."""
        self.model.train()

        f_batch = self._get_batch(self._forget_iter, self.forget_loader)
        f_mean, _ = self.model(
            f_batch["states"],
            f_batch["actions"],
            f_batch["returns_to_go"],
            f_batch["timesteps"],
            f_batch["attention_mask"],
        )
        forget_nll = gaussian_nll(
            f_mean,
            f_batch["actions"],
            self.model.action_log_var,
            f_batch["attention_mask"],
        )

        r_batch = self._get_batch(self._retain_iter, self.retain_loader)
        r_mean_unlearn, _ = self.model(
            r_batch["states"],
            r_batch["actions"],
            r_batch["returns_to_go"],
            r_batch["timesteps"],
            r_batch["attention_mask"],
        )
        with torch.no_grad():
            r_mean_base, _ = self.base_model(
                r_batch["states"],
                r_batch["actions"],
                r_batch["returns_to_go"],
                r_batch["timesteps"],
                r_batch["attention_mask"],
            )

        sigma_sq = torch.clamp(torch.exp(self.model.action_log_var), min=1e-4)
        kl_per_token = 0.5 * ((r_mean_base - r_mean_unlearn) ** 2 / sigma_sq).sum(
            dim=-1
        )
        mask = r_batch["attention_mask"]
        kl = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)

        loss = -forget_nll + self.kl_weight * kl

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.grad_clip,
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "forget_nll": forget_nll.item(),
            "kl": kl.item(),
        }

    def run_ascent(self, n_steps: int) -> list[dict]:
        """Run N steps of weighted gradient ascent."""
        metrics_log = []
        log_interval = max(1, n_steps // 10)

        for step in range(n_steps):
            metrics = self.ascent_step()
            metrics_log.append(metrics)

            if step % log_interval == 0 or step == n_steps - 1:
                pct = step / n_steps * 100
                print(
                    f"[W-Ascent {pct:5.1f}%] step {step}/{n_steps}  "
                    f"loss={metrics['loss']:.4f}  "
                    f"nll={metrics['forget_nll']:.4f}  "
                    f"kl={metrics['kl']:.6f}"
                )

        return metrics_log

    def refit_head(
        self,
        n_steps: int = 10000,
        lr: float = 1e-4,
        batch_size: int = 64,
        reinit_head: bool = True,
    ) -> list[dict]:
        """Head refit, reuses GradientAscentUnlearner logic."""
        print("\n--- Head Refit Phase ---")

        if reinit_head:
            self.model.predict_action_mean.apply(self.model._init_weights)

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.predict_action_mean.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(
            self.model.predict_action_mean.parameters(),
            lr=lr,
        )
        loader = DataLoader(
            self._retain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        loader_iter = iter(loader)

        metrics_log = []
        log_interval = max(1, n_steps // 10)
        warmup = min(1000, n_steps // 10)

        for step in range(n_steps):
            if step < warmup:
                cur_lr = lr * step / max(1, warmup)
            else:
                progress = (step - warmup) / max(1, n_steps - warmup)
                cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            action_mean, _ = self.model(
                batch["states"],
                batch["actions"],
                batch["returns_to_go"],
                batch["timesteps"],
                batch["attention_mask"],
            )
            loss = gaussian_nll(
                action_mean,
                batch["actions"],
                self.model.action_log_var,
                batch["attention_mask"],
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.predict_action_mean.parameters(),
                0.25,
            )
            optimizer.step()

            metrics = {"refit_loss": loss.item(), "refit_lr": cur_lr}
            metrics_log.append(metrics)

            if step % log_interval == 0 or step == n_steps - 1:
                pct = step / n_steps * 100
                print(
                    f"[Refit {pct:5.1f}%] step {step}/{n_steps}  "
                    f"loss={loss.item():.4f}  lr={cur_lr:.6f}"
                )

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.action_log_var.requires_grad = False

        return metrics_log


# ---------------------------------------------------------------------------
# Direction 2: Component-Selective Unlearning
# ---------------------------------------------------------------------------


def build_valid_selective_targets(n_layers: int) -> set[str]:
    targets = {
        "all",
        "attn",
        "mlp",
        "scan_layers",
        "scan_attn_layers",
        "scan_mlp_layers",
    }
    for idx in range(n_layers):
        layer_target = f"layer_{idx}"
        targets.add(layer_target)
        targets.add(f"attn.{layer_target}")
        targets.add(f"mlp.{layer_target}")
        targets.add(f"attn_{layer_target}")
        targets.add(f"mlp_{layer_target}")
    # Random-mask control targets (random parameter subset matching attn.layer_i parameter count)
    for idx in range(n_layers):
        targets.add(f"random_matched_attn_l{idx}")
    return targets


def _normalize_target(target: str) -> str:
    normalized = str(target).strip().lower()
    if normalized.startswith("attn_layer_"):
        return normalized.replace("attn_layer_", "attn.layer_", 1)
    if normalized.startswith("mlp_layer_"):
        return normalized.replace("mlp_layer_", "mlp.layer_", 1)
    # random_matched_attn_l0/1/2 keep as-is
    if normalized.startswith("random_matched_"):
        return normalized
    return normalized


def _parse_target(target: str, n_layers: int) -> tuple[str, int | None]:
    normalized = _normalize_target(target)
    valid_targets = build_valid_selective_targets(n_layers)
    if normalized not in valid_targets:
        raise ValueError(
            f"target_components={target!r} is invalid, valid options: {sorted(valid_targets)}"
        )

    # Special handling for Random-mask targets
    if normalized.startswith("random_matched_"):
        return normalized, None

    if normalized in {"all", "attn", "mlp"}:
        return normalized, None

    if normalized.startswith("layer_"):
        return "all", int(normalized.split("_", 1)[1])

    component, layer_token = normalized.split(".", 1)
    return component, int(layer_token.split("_", 1)[1])


def _select_body_params(
    model: DecisionTransformer,
    target: str,
    random_seed: int = 42,
) -> list[nn.Parameter]:
    """Return body parameters to unfreeze based on target.

    Always excludes action head and variance (consistent with existing GA).
    For random_matched_attn_lN target, randomly samples the same number of
    body parameters as attn.layer_N.
    """
    excluded_prefixes = ("predict_action_mean.", "action_log_var")
    component, layer_idx = _parse_target(target, len(model.blocks))

    # Random-mask control: match attn.layer_N parameter count
    if isinstance(component, str) and component.startswith("random_matched_"):
        # Parse the layer number to match
        match_layer = int(component.split("_l")[-1])
        ref_target = f"attn.layer_{match_layer}"
        ref_params = _select_body_params(model, ref_target)
        ref_n_params = sum(p.numel() for p in ref_params)

        # Collect all body parameters (excluding head and variance)
        all_body = [
            (n, p)
            for n, p in model.named_parameters()
            if not any(n.startswith(ex) for ex in excluded_prefixes)
        ]

        # Randomly sample by parameter tensor until total count >= ref_n_params
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(len(all_body))
        selected = []
        total = 0
        for idx in indices:
            _, param = all_body[idx]
            selected.append(param)
            total += param.numel()
            if total >= ref_n_params:
                break

        return selected

    if component == "all" and layer_idx is None:
        return [
            p
            for n, p in model.named_parameters()
            if not any(n.startswith(ex) for ex in excluded_prefixes)
        ]

    selected = []
    for name, param in model.named_parameters():
        if any(name.startswith(ex) for ex in excluded_prefixes):
            continue
        if layer_idx is not None and f"blocks.{layer_idx}." not in name:
            continue
        if component == "all":
            selected.append(param)
        elif component == "attn" and ".attn." in name:
            selected.append(param)
        elif component == "mlp" and ".mlp." in name:
            selected.append(param)

    return selected


def _select_body_named_params(
    model: DecisionTransformer,
    target: str,
) -> list[tuple[str, nn.Parameter]]:
    excluded_prefixes = ("predict_action_mean.", "action_log_var")
    component, layer_idx = _parse_target(target, len(model.blocks))

    if component == "all" and layer_idx is None:
        return [
            (n, p)
            for n, p in model.named_parameters()
            if not any(n.startswith(ex) for ex in excluded_prefixes)
        ]

    selected: list[tuple[str, nn.Parameter]] = []
    for name, param in model.named_parameters():
        if any(name.startswith(ex) for ex in excluded_prefixes):
            continue
        if layer_idx is not None and f"blocks.{layer_idx}." not in name:
            continue
        if component == "all":
            selected.append((name, param))
        elif component == "attn" and ".attn." in name:
            selected.append((name, param))
        elif component == "mlp" and ".mlp." in name:
            selected.append((name, param))
    return selected


def _compute_batch_nll(
    model: DecisionTransformer, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    action_mean, _ = model(
        batch["states"],
        batch["actions"],
        batch["returns_to_go"],
        batch["timesteps"],
        batch["attention_mask"],
    )
    return gaussian_nll(
        action_mean,
        batch["actions"],
        model.action_log_var,
        batch["attention_mask"],
    )


def _grad_norm(named_params: list[tuple[str, nn.Parameter]]) -> float:
    total = 0.0
    for _name, param in named_params:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


@torch.no_grad()
def _move_batch_to_device(
    batch: dict[str, torch.Tensor], device: str
) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def compute_pre_unlearning_localization(
    model: DecisionTransformer,
    forget_dataset: TrajectoryDataset,
    retain_dataset: TrajectoryDataset,
    *,
    batch_size: int = 64,
    max_batches: int = 2,
    device: str = "cuda",
) -> dict[str, object]:
    model = model.to(device)
    model.train()

    forget_loader = DataLoader(
        forget_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    targets = ["all", "attn", "mlp"]
    for layer_idx in range(len(model.blocks)):
        targets.append(f"attn.layer_{layer_idx}")
        targets.append(f"mlp.layer_{layer_idx}")

    named_targets = {
        target: _select_body_named_params(model, target) for target in targets
    }
    summary = {
        target: {
            "n_params": int(sum(param.numel() for _, param in named_params)),
            "forget_grad_norm_sum": 0.0,
            "retain_grad_norm_sum": 0.0,
            "batches": 0,
        }
        for target, named_params in named_targets.items()
    }

    processed = 0
    for forget_batch, retain_batch in zip(forget_loader, retain_loader):
        if processed >= max_batches:
            break
        forget_batch = _move_batch_to_device(forget_batch, device)
        retain_batch = _move_batch_to_device(retain_batch, device)

        model.zero_grad(set_to_none=True)
        forget_nll = _compute_batch_nll(model, forget_batch)
        forget_nll.backward()
        for target, named_params in named_targets.items():
            summary[target]["forget_grad_norm_sum"] += _grad_norm(named_params)

        model.zero_grad(set_to_none=True)
        retain_nll = _compute_batch_nll(model, retain_batch)
        retain_nll.backward()
        for target, named_params in named_targets.items():
            summary[target]["retain_grad_norm_sum"] += _grad_norm(named_params)
            summary[target]["batches"] += 1

        processed += 1

    model.zero_grad(set_to_none=True)

    per_target: dict[str, dict[str, float | int]] = {}
    for target, metrics in summary.items():
        batches = max(1, int(metrics["batches"]))
        forget_grad = float(metrics["forget_grad_norm_sum"]) / batches
        retain_grad = float(metrics["retain_grad_norm_sum"]) / batches
        grad_ratio = forget_grad / max(retain_grad, 1e-12)
        per_target[target] = {
            "n_params": int(metrics["n_params"]),
            "forget_grad_norm": forget_grad,
            "retain_grad_norm": retain_grad,
            "forget_retain_grad_ratio": grad_ratio,
            "log_forget_retain_grad_ratio": float(math.log(max(grad_ratio, 1e-12))),
        }

    ranked = sorted(
        per_target.items(),
        key=lambda item: item[1]["forget_retain_grad_ratio"],
        reverse=True,
    )
    top_target = ranked[0][0] if ranked else None

    return {
        "metric": "gradient_norm_ratio",
        "max_batches": int(processed),
        "top_target": top_target,
        "targets": per_target,
    }


class SelectiveGradientAscentUnlearner:
    """Component-Selective GA + KL + Head Refit.

    Identical GA+KL+Refit flow as GradientAscentUnlearner,
    but only performs GA on specified components (attention / MLP / specific layers),
    with the remaining body parameters frozen.

    target_components:
        "attn"   — only modify attention (qkv, proj) parameters
        "mlp"    — only modify MLP parameters
        "all"    — modify all body (= uniform baseline)
        "layer_0/1/2" — only modify all parameters in the specified layer
    """

    def __init__(
        self,
        model: DecisionTransformer,
        base_model: DecisionTransformer,
        forget_dataset: TrajectoryDataset,
        retain_dataset: TrajectoryDataset,
        target_components: str = "attn",
        kl_weight: float = 1.0,
        lr: float = 1e-4,
        grad_clip: float = 0.25,
        batch_size: int = 64,
        device: str = "cuda",
        random_seed: int = 42,
    ):
        self.target_components = _normalize_target(target_components)
        self.model = model.to(device)
        self.device = device
        self.kl_weight = kl_weight
        self.grad_clip = grad_clip

        # Base model frozen
        self.base_model = base_model.to(device)
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

        # Freeze all parameters, then selectively unfreeze
        for p in self.model.parameters():
            p.requires_grad = False

        target_params = _select_body_params(
            self.model, self.target_components, random_seed=random_seed
        )
        if not target_params:
            raise ValueError(
                f"target_components={target_components!r} matched no trainable parameters"
            )
        for p in target_params:
            p.requires_grad = True

        # Statistics
        n_target = sum(p.numel() for p in target_params)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(
            f"[Selective GA] target={self.target_components}, "
            f"trainable params: {n_target:,} / {n_total:,} ({n_target / n_total * 100:.1f}%)"
        )

        self.optimizer = torch.optim.Adam(target_params, lr=lr)

        self.forget_loader = DataLoader(
            forget_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self.retain_loader = DataLoader(
            retain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        self._forget_iter = iter(self.forget_loader)
        self._retain_iter = iter(self.retain_loader)
        self._retain_dataset = retain_dataset

    def _get_batch(self, iterator, loader) -> dict:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
            if loader is self.forget_loader:
                self._forget_iter = iterator
            else:
                self._retain_iter = iterator
        return {k: v.to(self.device) for k, v in batch.items()}

    def ascent_step(self) -> dict:
        """Same as GradientAscentUnlearner.ascent_step."""
        self.model.train()

        f_batch = self._get_batch(self._forget_iter, self.forget_loader)
        f_mean, _ = self.model(
            f_batch["states"],
            f_batch["actions"],
            f_batch["returns_to_go"],
            f_batch["timesteps"],
            f_batch["attention_mask"],
        )
        forget_nll = gaussian_nll(
            f_mean,
            f_batch["actions"],
            self.model.action_log_var,
            f_batch["attention_mask"],
        )

        r_batch = self._get_batch(self._retain_iter, self.retain_loader)
        r_mean_unlearn, _ = self.model(
            r_batch["states"],
            r_batch["actions"],
            r_batch["returns_to_go"],
            r_batch["timesteps"],
            r_batch["attention_mask"],
        )
        with torch.no_grad():
            r_mean_base, _ = self.base_model(
                r_batch["states"],
                r_batch["actions"],
                r_batch["returns_to_go"],
                r_batch["timesteps"],
                r_batch["attention_mask"],
            )

        sigma_sq = torch.clamp(torch.exp(self.model.action_log_var), min=1e-4)
        kl_per_token = 0.5 * ((r_mean_base - r_mean_unlearn) ** 2 / sigma_sq).sum(
            dim=-1
        )
        mask = r_batch["attention_mask"]
        kl = (kl_per_token * mask).sum() / mask.sum().clamp(min=1)

        loss = -forget_nll + self.kl_weight * kl

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.grad_clip,
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "forget_nll": forget_nll.item(),
            "kl": kl.item(),
        }

    def run_ascent(self, n_steps: int) -> list[dict]:
        """Run N steps of selective gradient ascent."""
        metrics_log = []
        log_interval = max(1, n_steps // 10)

        for step in range(n_steps):
            metrics = self.ascent_step()
            metrics_log.append(metrics)

            if step % log_interval == 0 or step == n_steps - 1:
                pct = step / n_steps * 100
                print(
                    f"[S-GA({self.target_components}) {pct:5.1f}%] "
                    f"step {step}/{n_steps}  "
                    f"loss={metrics['loss']:.4f}  "
                    f"nll={metrics['forget_nll']:.4f}  "
                    f"kl={metrics['kl']:.6f}"
                )

        return metrics_log

    def refit_head(
        self,
        n_steps: int = 10000,
        lr: float = 1e-4,
        batch_size: int = 64,
        reinit_head: bool = True,
    ) -> list[dict]:
        """Head refit (same as GradientAscentUnlearner)."""
        print("\n--- Head Refit Phase ---")

        if reinit_head:
            self.model.predict_action_mean.apply(self.model._init_weights)

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.predict_action_mean.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(
            self.model.predict_action_mean.parameters(),
            lr=lr,
        )
        loader = DataLoader(
            self._retain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        loader_iter = iter(loader)

        metrics_log = []
        log_interval = max(1, n_steps // 10)
        warmup = min(1000, n_steps // 10)

        for step in range(n_steps):
            if step < warmup:
                cur_lr = lr * step / max(1, warmup)
            else:
                progress = (step - warmup) / max(1, n_steps - warmup)
                cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            action_mean, _ = self.model(
                batch["states"],
                batch["actions"],
                batch["returns_to_go"],
                batch["timesteps"],
                batch["attention_mask"],
            )
            loss = gaussian_nll(
                action_mean,
                batch["actions"],
                self.model.action_log_var,
                batch["attention_mask"],
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.predict_action_mean.parameters(),
                0.25,
            )
            optimizer.step()

            metrics = {"refit_loss": loss.item(), "refit_lr": cur_lr}
            metrics_log.append(metrics)

            if step % log_interval == 0 or step == n_steps - 1:
                pct = step / n_steps * 100
                print(
                    f"[Refit {pct:5.1f}%] step {step}/{n_steps}  "
                    f"loss={loss.item():.4f}  lr={cur_lr:.6f}"
                )

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.action_log_var.requires_grad = False

        return metrics_log


class FisherUnlearner:
    """Influence-based unlearning using diagonal Fisher Information approximation.

    Simplified version of TrajDeleter (Liu et al., 2025):
    θ' = θ + scale * H^{-1} ∇_θ L(D_f; θ)

    where H is the diagonal FIM approximation on D_r, and L is the NLL loss.
    Intuition: scale the forget set gradients using retain set curvature,
    so parameter updates mainly occur in directions with low impact on retain.

    Differences from full TrajDeleter:
    - Uses diagonal FIM (instead of block-diagonal / KFAC)
    - Does not use Hessian-vector product iterative solving
    - Single-step Newton update (instead of multi-step influence estimation)
    """

    def __init__(
        self,
        model: DecisionTransformer,
        forget_dataset: TrajectoryDataset,
        retain_dataset: TrajectoryDataset,
        damping: float = 1e-3,
        scale: float = 1.0,
        batch_size: int = 64,
        n_fisher_samples: int = 50,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.damping = damping
        self.scale = scale
        self.n_fisher_samples = n_fisher_samples

        self._forget_dataset = forget_dataset
        self._retain_dataset = retain_dataset
        self.batch_size = batch_size

        # Get body parameter names (excluding action head and variance)
        self._body_names = _get_body_param_names(model)

    def _get_body_params(self) -> list[tuple[str, nn.Parameter]]:
        """Return body parameters as (name, param) list."""
        return [
            (n, p) for n, p in self.model.named_parameters() if n in self._body_names
        ]

    def _compute_fisher_diagonal(self) -> dict[str, torch.Tensor]:
        """Compute diagonal FIM approximation on the retain set.

        F_ii = E_{x~D_r}[ (∂L/∂θ_i)^2 ]
        """
        self.model.eval()
        fisher = {}
        for n, p in self._get_body_params():
            fisher[n] = torch.zeros_like(p.data)

        loader = DataLoader(
            self._retain_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        loader_iter = iter(loader)

        for _ in range(self.n_fisher_samples):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.model.zero_grad()
            mean, _ = self.model(
                batch["states"],
                batch["actions"],
                batch["returns_to_go"],
                batch["timesteps"],
                batch["attention_mask"],
            )
            nll = gaussian_nll(
                mean,
                batch["actions"],
                self.model.action_log_var,
                batch["attention_mask"],
            )
            nll.backward()

            for n, p in self._get_body_params():
                if p.grad is not None:
                    fisher[n] += p.grad.data**2

        # Normalize
        for n in fisher:
            fisher[n] /= self.n_fisher_samples

        return fisher

    def _compute_forget_gradient(self) -> dict[str, torch.Tensor]:
        """Compute the average gradient on the forget set."""
        self.model.eval()
        grad_sum = {}
        for n, p in self._get_body_params():
            grad_sum[n] = torch.zeros_like(p.data)

        loader = DataLoader(
            self._forget_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

        n_batches = 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.model.zero_grad()
            mean, _ = self.model(
                batch["states"],
                batch["actions"],
                batch["returns_to_go"],
                batch["timesteps"],
                batch["attention_mask"],
            )
            nll = gaussian_nll(
                mean,
                batch["actions"],
                self.model.action_log_var,
                batch["attention_mask"],
            )
            nll.backward()

            for n, p in self._get_body_params():
                if p.grad is not None:
                    grad_sum[n] += p.grad.data
            n_batches += 1

        for n in grad_sum:
            grad_sum[n] /= max(n_batches, 1)

        return grad_sum

    def run_unlearning(self) -> dict:
        """Execute Fisher-based unlearning: θ' = θ + scale * F^{-1} ∇L_f.

        The positive sign is because ∇L_forget points in the direction of
        increasing loss (i.e., "anti-learning"). Returns a diagnostics dict.
        """
        print("[Fisher Unlearning] Computing Fisher diagonal on retain set...")
        fisher = self._compute_fisher_diagonal()

        print("[Fisher Unlearning] Computing gradient on forget set...")
        forget_grad = self._compute_forget_gradient()

        # Newton update: θ' = θ + scale * (F + damping * I)^{-1} * ∇L_forget
        print("[Fisher Unlearning] Performing Newton update...")
        update_norms = {}
        with torch.no_grad():
            for n, p in self._get_body_params():
                f_diag = fisher[n] + self.damping
                update = self.scale * forget_grad[n] / f_diag
                p.data += update
                update_norms[n] = update.norm().item()

        total_update_norm = sum(v**2 for v in update_norms.values()) ** 0.5
        print(f"[Fisher Unlearning] Done. Total update norm: {total_update_norm:.6f}")

        return {
            "method": "fisher_unlearning",
            "damping": self.damping,
            "scale": self.scale,
            "n_fisher_samples": self.n_fisher_samples,
            "total_update_norm": total_update_norm,
        }

    def refit_head(
        self,
        n_steps: int = 10000,
        lr: float = 1e-4,
        batch_size: int = 64,
    ) -> list[dict]:
        """Head refit: re-initialize the action head and retrain on the retain set."""
        print("\n--- Head Refit Phase ---")

        self.model.predict_action_mean.apply(self.model._init_weights)

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.predict_action_mean.parameters():
            p.requires_grad = True

        optimizer = torch.optim.Adam(
            self.model.predict_action_mean.parameters(),
            lr=lr,
        )
        loader = DataLoader(
            self._retain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        loader_iter = iter(loader)

        metrics_log = []
        log_interval = max(1, n_steps // 10)

        for step in range(n_steps):
            self.model.train()
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            mean, _ = self.model(
                batch["states"],
                batch["actions"],
                batch["returns_to_go"],
                batch["timesteps"],
                batch["attention_mask"],
            )
            loss = gaussian_nll(
                mean,
                batch["actions"],
                self.model.action_log_var,
                batch["attention_mask"],
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.predict_action_mean.parameters(),
                0.25,
            )
            optimizer.step()

            metrics = {"refit_loss": loss.item()}
            metrics_log.append(metrics)

            if step % log_interval == 0 or step == n_steps - 1:
                pct = step / n_steps * 100
                print(
                    f"[Refit {pct:5.1f}%] step {step}/{n_steps}  loss={loss.item():.4f}"
                )

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.action_log_var.requires_grad = False

        return metrics_log
