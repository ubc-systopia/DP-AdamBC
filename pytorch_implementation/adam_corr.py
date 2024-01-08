import os.path
import numpy
import torch
# from torch.optim import _functional as F
import math
import torch
from torch import Tensor
from typing import List, Optional
from torch.optim.optimizer import Optimizer
import pickle
import wandb
import numpy as np
# from torch.optim.optimizer import (Optimizer, required, _use_grad_for_differentiable, _default_to_fused_or_foreach,
#                         _differentiable_doc, _foreach_doc, _maximize_doc)
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype


def _tmp_get_summary_stats(vec):
    return {
        'min': torch.min(vec), 'max': torch.max(vec),
        'mean': torch.mean(vec), 'median': torch.quantile(vec, 0.50),
        'q1': torch.quantile(vec, 0.25), 'q3': torch.quantile(vec, 0.75)
    }


def adam_corr(params: List[Tensor],
              grads: List[Tensor],
              mean_clipped_grads: List[Tensor],
              exp_avgs: List[Tensor],
              exp_avg_sqs: List[Tensor],
              exp_avgs_clean: List[Tensor],
              exp_avg_sqs_clean: List[Tensor],
              max_exp_avg_sqs: List[Tensor],
              state_steps: List[int],
              *,
              amsgrad: bool,
              beta1: float,
              beta2: float,
              lr: float,
              weight_decay: float,
              eps: float,
              dp_batch_size,
              dp_noise_multiplier,
              dp_l2_norm_clip,
              eps_root):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    perc_corr = 0
    perc_zero = 0
    dummy_num_param_count = 0

    for i, param in enumerate(params):
        grad = grads[i]
        mean_clipped_grad = mean_clipped_grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_clean = exp_avgs_clean[i]
        exp_avg_sq_clean = exp_avg_sqs_clean[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        exp_avg_clean.mul_(beta1).add_(mean_clipped_grad, alpha=1 - beta1)
        exp_avg_sq_clean.mul_(beta2).addcmul_(mean_clipped_grad, mean_clipped_grad.conj(), value=1 - beta2)

        # corr for noise variance
        exp_avg_sq_hat = torch.divide(exp_avg_sq, bias_correction2)
        noise_err = (1 / dp_batch_size ** 2) * dp_noise_multiplier ** 2 * dp_l2_norm_clip ** 2
        tmp_exp_avg_sq = torch.subtract(exp_avg_sq_hat, noise_err)
        eps_vec = eps_root * torch.ones_like(exp_avg_sq)

        # Ablation: for subtracting a different Phi
        # tmp_exp_avg_sq = torch.subtract(exp_avg_sq, eps_root)
        # eps_vec = (3e-14) * torch.ones_like(exp_avg_sq)

        # 1- replace small values with eps_root
        exp_avg_sq_corr = torch.maximum(tmp_exp_avg_sq, eps_vec)
        # 2- only making the negative ones replaced by eps_root
        # tmp_mask = (tmp_exp_avg_sq > 0).int()
        # exp_avg_sq_corr = torch.add(torch.multiply((1 - tmp_mask), eps_vec), torch.multiply(tmp_mask, tmp_exp_avg_sq))
        # 3- max(v_corr, m_t^2) + gamma
        # exp_avg_sq_corr = torch.add(torch.maximum(torch.divide(tmp_exp_avg_sq, bias_correction2),
        #                                           torch.square(torch.divide(exp_avg, bias_correction1))),
        #                             eps_vec)
        # 4- Ablation: sqrt(v_corr + gamma)
        # eps_vec2 = 1e-20 * torch.ones_like(tmp_exp_avg_sq)
        # exp_avg_sq_corr = torch.maximum(torch.add(tmp_exp_avg_sq, eps_vec), eps_vec2)
        # exp_avg_sq_corr = torch.add(torch.maximum(tmp_exp_avg_sq, torch.zeros_like(tmp_exp_avg_sq)), eps_vec)

        tmp_vec = tmp_exp_avg_sq > 0  # params with corrected v_t values used
        perc_corr += torch.sum(tmp_vec)
        # perc_corr += torch.tensor(0)
        # tmp_vec2 = torch.maximum(tmp_exp_avg_sq,
        #                          torch.square(exp_avg)) > eps_vec  # params with corrected vt larger than gamma
        tmp_vec2 = tmp_exp_avg_sq > eps_vec
        perc_zero += torch.sum(tmp_vec2)
        dummy_num_param_count += tmp_vec2.flatten().shape[0]

        # ablation study
        # tmp_err experiment of subtracting random number as error
        # tmp_exp_avg_sq = torch.subtract(exp_avg_sq, eps_root)
        # eps_vec = eps_root * torch.ones_like(exp_avg_sq)
        # # replace small values with eps_root
        # # exp_avg_sq_corr = torch.maximum(tmp_exp_avg_sq, eps_vec)
        # # only making the negative ones replaced by eps_root
        # tmp_mask = (tmp_exp_avg_sq > 0).int()
        # exp_avg_sq_corr = torch.add(torch.multiply((1 - tmp_mask), eps_vec), torch.multiply(tmp_mask, tmp_exp_avg_sq))

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            # torch.maximum(max_exp_avg_sqs[i], exp_avg_sq_corr, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            # denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            raise NotImplementedError
        else:
            # 1- or 2-
            # denom = (exp_avg_sq_corr.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            # denom = (exp_avg_sq_corr.sqrt() / math.sqrt(bias_correction2))
            denom = exp_avg_sq_corr.sqrt()
            # 3-
            # denom = exp_avg_sq_corr.sqrt()
            # 4- replace negative values with 0
            # denom = (torch.nan_to_num_(exp_avg_sq_corr.sqrt(), 0) / math.sqrt(bias_correction2)).add_(eps)

        # 1- or 2-
        step_size = lr / bias_correction1
        param.addcdiv_(exp_avg, denom, value=-step_size)
        # # 3-
        # step_size = lr
        # nume = torch.divide(exp_avg, bias_correction1)
        # param.addcdiv_(nume, denom, value=-step_size)

    mt_norm, vt_norm, vt_corr_norm, mt_clean_norm, vt_clean_norm = [torch.nan] * 5
    hist_dict = {}
    summary_stats_dict = {}

    perc_corr_total = perc_corr / dummy_num_param_count
    perc_zero_total = perc_zero / dummy_num_param_count

    return mt_norm, vt_norm, vt_corr_norm, mt_clean_norm, vt_clean_norm, \
           perc_corr_total, perc_zero_total, hist_dict, summary_stats_dict, step


class AdamCorr(Optimizer):
    """Modified from torch's version of Adam"""

    def __init__(self, params, dp_batch_size, dp_noise_multiplier, dp_l2_norm_clip, eps_root,
                 lr=1e-3, betas=(0.9, 0.999), eps=1e-8, gamma_decay=1,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamCorr, self).__init__(params, defaults)
        self.dp_batch_size = dp_batch_size
        self.dp_noise_multiplier = dp_noise_multiplier
        self.dp_l2_norm_clip = dp_l2_norm_clip
        self.eps_root = eps_root
        self.gamma_decay = gamma_decay

    def __setstate__(self, state):
        super(AdamCorr, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []  # private grad: mean clipped noised grad
            mean_clipped_grads = []  # mean clipped grad (unnoised)
            exp_avgs = []  # moving average of priv grad
            exp_avg_sqs = []  # moving average of priv grad squared
            exp_avgs_clean = []  # moving average of mean clipped grad
            exp_avg_sqs_clean = []  # moving average of mean clipped grad squared
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            cur_gamma = self.eps_root

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)
                    mean_clipped_grad = p.summed_grad / self.dp_batch_size
                    mean_clipped_grads.append(mean_clipped_grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_clean'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq_clean'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    exp_avgs_clean.append(state['exp_avg_clean'])
                    exp_avg_sqs_clean.append(state['exp_avg_sq_clean'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                        raise NotImplementedError

                    # gamma scheduler
                    cur_gamma = self.eps_root * (self.gamma_decay ** state['step'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            # use adam_corr
            mt_norm, vt_norm, vt_corr_norm, mt_clean_norm, vt_clean_norm, \
            mean_perc_corr, mean_perc_zero, hist_dict, summary_stats_dict, dummy_step = \
                adam_corr(params_with_grad,
                          grads,
                          mean_clipped_grads,
                          exp_avgs,
                          exp_avg_sqs,
                          exp_avgs_clean,
                          exp_avg_sqs_clean,
                          max_exp_avg_sqs,
                          state_steps,
                          amsgrad=group['amsgrad'],
                          beta1=beta1,
                          beta2=beta2,
                          lr=group['lr'],
                          weight_decay=group['weight_decay'],
                          eps=group['eps'],
                          dp_batch_size=self.dp_batch_size,
                          # dp_batch_size=actural_batch_size,  # if used Possion sampling
                          dp_noise_multiplier=self.dp_noise_multiplier,
                          dp_l2_norm_clip=self.dp_l2_norm_clip,
                          # eps_root=self.eps_root,
                          eps_root=cur_gamma,
                          )
        return loss, {
            'mt_norm': mt_norm, "vt_norm": vt_norm, 'vt_corr_norm': vt_corr_norm, 'mt_clean_norm': mt_clean_norm,
            'vt_clean_norm': vt_clean_norm, 'mean_perc_corr': mean_perc_corr, 'mean_perc_zero': mean_perc_zero,
        }, hist_dict, summary_stats_dict, dummy_step, cur_gamma


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         tmp_err: float, ):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    mt_norms = []
    vt_norms = []
    tmp_mt_clean_grads = []
    tmp_vt_clean_grads = []
    tmp_snr = []

    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        # ablation study
        # # tmp_err: effect of adding constant bias in second moment
        # exp_avg_sq = torch.add(exp_avg_sq, tmp_err)
        # tmp_exp_avg_sq = torch.subtract(exp_avg_sq, tmp_err)
        # eps_vec = tmp_err * torch.ones_like(exp_avg_sq)
        # # replace small values with eps_root
        # # exp_avg_sq_corr = torch.maximum(tmp_exp_avg_sq, eps_vec)
        # # only making the negative ones replaced by eps_root
        # tmp_mask = (tmp_exp_avg_sq > 0).int()
        # exp_avg_sq_corr = torch.add(torch.multiply((1 - tmp_mask), eps_vec), torch.multiply(tmp_mask, tmp_exp_avg_sq))

        mt = torch.divide(exp_avg, bias_correction1)
        vt = torch.divide(exp_avg_sq, bias_correction2)
        mt_norms.append(torch.linalg.norm(mt))
        vt_norms.append(torch.linalg.norm(vt))

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)

    mt_norm = torch.linalg.norm(torch.stack(mt_norms))
    vt_norm = torch.linalg.norm(torch.stack(vt_norms))
    hist_dict = {}
    summary_stats_dict = {}

    return mt_norm, vt_norm, hist_dict, summary_stats_dict, step


class OrigAdam(Optimizer):
    """the copy from torch"""
    def __init__(self, params, lr=1e-3, betas=(0.5, 0.5), eps=1e-8,
                 weight_decay=0, amsgrad=False, tmp_err=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(OrigAdam, self).__init__(params, defaults)
        self.tmp_err = tmp_err

    def __setstate__(self, state):
        super(OrigAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            mt_norm, vt_norm, hist_dict, summary_stats_dict, dummy_step = \
                adam(params_with_grad,
                     grads,
                     exp_avgs,
                     exp_avg_sqs,
                     max_exp_avg_sqs,
                     state_steps,
                     amsgrad=group['amsgrad'],
                     beta1=beta1,
                     beta2=beta2,
                     lr=group['lr'],
                     weight_decay=group['weight_decay'],
                     eps=group['eps'],
                     tmp_err=self.tmp_err)

        return loss, {
            'mt_norm': mt_norm, 'vt_norm': vt_norm,
        }, hist_dict, summary_stats_dict, dummy_step


def my_sgdm(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         tmp_err: float, ):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    for i, param in enumerate(params):

        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1)
        exp_avg_sq.mul_(0)

        step_size = lr / bias_correction1

        # param.addcdiv_(exp_avg, denom, value=-step_size)
        param.add_(exp_avg, alpha=-step_size)


class MySGDM(Optimizer):
    """the copy from torch"""
    def __init__(self, params, lr=1e-3, betas=(0.5, 0.5), eps=1e-8,
                 weight_decay=0, amsgrad=False, tmp_err=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(MySGDM, self).__init__(params, defaults)
        self.tmp_err = tmp_err

    def __setstate__(self, state):
        super(MySGDM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            my_sgdm(params_with_grad,
                 grads,
                 exp_avgs,
                 exp_avg_sqs,
                 max_exp_avg_sqs,
                 state_steps,
                 amsgrad=group['amsgrad'],
                 beta1=beta1,
                 beta2=beta2,
                 lr=group['lr'],
                 weight_decay=group['weight_decay'],
                 eps=group['eps'],
                 tmp_err=self.tmp_err)

        return loss
