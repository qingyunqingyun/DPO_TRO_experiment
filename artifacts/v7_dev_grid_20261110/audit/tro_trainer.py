"""OpenRLHF trainer for soft-label, label-DRO, and per-pair TRO DPO."""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from drdpo_flash_attn_compat import install_flash_attn_distributed_fallback

install_flash_attn_distributed_fallback()

from openrlhf.trainer import DPOTrainer
from openrlhf.utils.distributed_sampler import DistributedSampler

from tro_math import dro_target, tro_dpo_loss


class TROTrainer(DPOTrainer):
    """DPOTrainer variant that consumes reliability metadata per pair."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tro_method = getattr(self.args, "tro_method", "tro_common")

    def _metadata(self, extras, device):
        def t(name, default=0.0):
            return torch.tensor([float(x.get(name, default)) for x in extras], dtype=torch.float32, device=device)

        return t("p_hat", 0.5), t("lower", 0.5), t("upper", 0.5), t("lambda_i", 0.0), t("p_star", 0.5), t("n_votes", 0.0)

    def _loss(self, chosen_logps, rejected_logps, ref_chosen, ref_rejected, extras):
        p_hat, lower, upper, lam, p_star, n_votes = self._metadata(extras, chosen_logps.device)
        method = self.tro_method
        if method == "hard_dpo":
            target = torch.ones_like(p_hat)
            per_pair, z = tro_dpo_loss(chosen_logps, rejected_logps, ref_chosen, ref_rejected, self.beta, target, target, target, torch.zeros_like(target))[:2]
            worst = target
        elif method == "soft_dpo":
            per_pair, z = tro_dpo_loss(chosen_logps, rejected_logps, ref_chosen, ref_rejected, self.beta, p_hat, p_hat, p_hat, torch.zeros_like(p_hat))[:2]
            target, worst = p_hat, p_hat
        elif method == "oracle_soft_dpo":
            per_pair, z = tro_dpo_loss(chosen_logps, rejected_logps, ref_chosen, ref_rejected, self.beta, p_star, p_star, p_star, torch.zeros_like(p_star))[:2]
            target, worst = p_star, p_star
        elif method in {"label_dro_common", "label_dro_n"}:
            _, z = tro_dpo_loss(chosen_logps, rejected_logps, ref_chosen, ref_rejected, self.beta, p_hat, lower, upper, torch.ones_like(p_hat))[:2]
            worst = dro_target(z, lower, upper, p_hat)
            target = worst
            per_pair = -target * F.logsigmoid(z) - (1.0 - target) * F.logsigmoid(-z)
        elif method in {"tro_common", "tro_trust_common", "tro_n", "fixed_lambda_tro", "shuffled_lambda_input_n"}:
            if method == "fixed_lambda_tro":
                lam = torch.full_like(lam, float(getattr(self.args, "tro_fixed_lambda", 0.5)))
            if method == "shuffled_lambda_input_n":
                # The dataset builder supplies the shuffled n-derived lambda.
                pass
            per_pair, z, target, worst = tro_dpo_loss(chosen_logps, rejected_logps, ref_chosen, ref_rejected, self.beta, p_hat, lower, upper, lam)
        else:
            raise ValueError(f"unknown tro_method={method}")
        nominal = -p_hat * F.logsigmoid(z) - (1.0 - p_hat) * F.logsigmoid(-z)
        return per_pair, z, target, worst, n_votes, nominal

    def _forward_pair(self, data):
        chosen_ids, c_mask, reject_ids, r_mask, extras = data
        chosen_ids = chosen_ids.squeeze(1).to(torch.cuda.current_device())
        c_mask = c_mask.squeeze(1).to(torch.cuda.current_device())
        reject_ids = reject_ids.squeeze(1).to(torch.cuda.current_device())
        r_mask = r_mask.squeeze(1).to(torch.cuda.current_device())
        chosen_logps, rejected_logps, aux_loss, nll_loss = self.concatenated_forward(
            self.model, chosen_ids, c_mask, reject_ids, r_mask, [x["prompt_id_len"] for x in extras]
        )
        with torch.no_grad():
            ref_chosen, ref_rejected, _, _ = self.concatenated_forward(
                self.ref_model, chosen_ids, c_mask, reject_ids, r_mask, [x["prompt_id_len"] for x in extras]
            )
        return chosen_logps, rejected_logps, ref_chosen, ref_rejected, aux_loss, nll_loss, extras

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if self.packing_samples:
            raise NotImplementedError("TRO first implementation uses unpacked samples for auditable masks")
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")
        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)
        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        loss_sum = acc_sum = robust_sum = margin_sum = lambda_sum = 0.0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples)
            step_bar = tqdm(range(len(self.train_dataloader)), desc=f"Train step of epoch {epoch}", disable=not self.strategy.is_rank_0())
            self.model.train(); self.ref_model.eval()
            for data in self.train_dataloader:
                chosen, rejected, ref_chosen, ref_rejected, aux, nll, extras = self._forward_pair(data)
                per_pair, z, target, worst, n_votes, nominal = self._loss(chosen, rejected, ref_chosen, ref_rejected, extras)
                preference_loss = per_pair.mean()
                robust_increment = (per_pair - nominal).mean()
                if not self.aux_loss: aux = 0
                if not self.nll_loss: nll = 0
                loss = preference_loss + aux * self.args.aux_loss_coef + nll * self.args.nll_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                acc = (self.beta * (chosen - ref_chosen) > self.beta * (rejected - ref_rejected)).float().mean().item()
                loss_sum += preference_loss.item(); acc_sum += acc; robust_sum += robust_increment.item()
                margin_sum += z.detach().abs().mean().item(); lambda_sum += float(target.detach().mean().item())
                logs = self.strategy.all_reduce({"loss": preference_loss.item(), "acc": acc, "target_mean": target.mean().item(), "margin_abs": z.detach().abs().mean().item(), "lr": self.scheduler.get_last_lr()[0]})
                step_bar.set_postfix(logs); step_bar.update()
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    logs.update({"loss_mean": loss_sum / self.strategy.accumulated_gradient, "robust_increment": robust_sum / self.strategy.accumulated_gradient, "target_mean_accum": lambda_sum / self.strategy.accumulated_gradient})
                    loss_sum = acc_sum = robust_sum = margin_sum = lambda_sum = 0.0
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs, {"consumed_samples": global_step * args.train_batch_size})
                step += 1
            epoch_bar.update()
        # A final HF-format export is useful for downstream generation
        # evaluation when no periodic save step was requested.  Keep this
        # behind the existing flag so ordinary training behavior is unchanged.
        if self.save_hf_ckpt and self.strategy.is_rank_0():
            final_path = os.path.join(args.ckpt_path, "final_hf")
            self.strategy.save_model(self.model, self.tokenizer, final_path)
        if self._wandb is not None and self.strategy.is_rank_0(): self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0(): self._tensorboard.close()

    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval(); loss_sum = acc_sum = times = 0.0
        bar = tqdm(range(len(eval_dataloader)), desc=f"Eval stage of global_step {steps}", disable=not self.strategy.is_rank_0())
        with torch.no_grad():
            for data in eval_dataloader:
                chosen, rejected, ref_chosen, ref_rejected, aux, nll, extras = self._forward_pair(data)
                per_pair, z, target, worst, _, _ = self._loss(chosen, rejected, ref_chosen, ref_rejected, extras)
                loss_sum += per_pair.mean().item()
                acc_sum += (self.beta * (chosen - ref_chosen) > self.beta * (rejected - ref_rejected)).float().mean().item()
                times += 1; bar.update()
        logs = self.strategy.all_reduce({"eval_loss": loss_sum / max(times, 1), "acc_mean": acc_sum / max(times, 1)})
        bar.set_postfix(logs)
        if self.strategy.is_rank_0():
            if self._wandb is not None: self._wandb.log({f"eval/{k}": v for k, v in {**logs, "global_step": steps}.items()})
            elif self._tensorboard is not None:
                for k, v in logs.items(): self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()
