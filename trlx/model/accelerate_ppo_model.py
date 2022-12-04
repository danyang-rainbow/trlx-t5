from typing import Tuple

import torch
from torchtyping import TensorType

from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLBatch
from trlx.model import register_model
from trlx.model.accelerate_base_model import AccelerateRLModel
from trlx.model.nn.ppo_models import (
    AdaptiveKLController,
    FixedKLController,
    T5HeadWithValueModel,
)
from trlx.pipeline.ppo_pipeline import PPORolloutStorage
from trlx.utils.modeling import logprobs_from_logits

def shift_tokens_right(input_ids, pad_token_id=0, decoder_start_token_id=0):
    """平移Label ids, 得到Deocde input ids
    """
    shifted_input_ids = torch.zeros_like(input_ids).to(input_ids.device)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids[shifted_input_ids==-100] = pad_token_id
    return shifted_input_ids

@register_model
class AcceleratePPOModel(AccelerateRLModel):
    def __init__(self, config):
        super().__init__(config)

        self.store = PPORolloutStorage(0)

        rollout_loader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        self.model, self.opt, self.scheduler, rollout_loader = self.accelerator.prepare(
            self.model, self.opt, self.scheduler, rollout_loader
        )

        self.store.clear_history()
        if config.method.target is not None:
            self.kl_ctl = AdaptiveKLController(
                config.method.init_kl_coef, config.method.target, config.method.horizon
            )
        else:
            self.kl_ctl = FixedKLController(config.method.init_kl_coef)

        self.generate_kwargs = dict(
            config.method.gen_kwargs,
            eos_token_id=1,
            pad_token_id=0,
        )

    def get_arch(self, config: TRLConfig):
        return T5HeadWithValueModel(
            self.config.model.model_path
        )
        
        # gpt self.config.model.model_path, self.config.model.num_layers_unfrozen

    def get_model_inputs(
        self,
        query_tensors: TensorType["batch_size", "query_size"],
        response_tensors: TensorType["batch_size", "response_size"],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        input_seq = query_tensors
        new_label_ids = response_tensors
        decoder_input_ids = shift_tokens_right(new_label_ids)
        
        
        print(input_seq.device)
        print(new_label_ids.device)
        print(decoder_input_ids.device)
        return input_seq, new_label_ids, decoder_input_ids
        

    def loss(self, batch: PPORLBatch):
        # Move `batch` data to `accelerator` device
        query_tensors = batch.query_tensors.to(self.accelerator.device)
        response_tensors = batch.response_tensors.to(self.accelerator.device)
        old_logprobs = batch.logprobs.to(self.accelerator.device)
        old_values = batch.values.to(self.accelerator.device)
        old_rewards = batch.rewards.to(self.accelerator.device)

        response_length = response_tensors.shape[-1]
        advantages, returns = self.config.method.get_advantages_and_returns(
            old_values, old_rewards, response_length
        )

        input_ids, labels, decoder_input_ids = self.get_model_inputs(
            query_tensors, response_tensors
        )
        
        model_inputs = {
            'input_ids': input_ids,
            'labels': labels,
            'decoder_input_ids': decoder_input_ids,
            "return_dict":True
        }
        
        model_outputs = self.model(
            **model_inputs
        )
        logits = model_outputs.logits
        values_pred = model_outputs.value
        logprobs = logprobs_from_logits(logits, labels)
        # Only the response part of the values/logprobs is needed
        
        attention_mask = torch.ones_like(logprobs, dtype=torch.long)
        logprobs, values_pred, mask = (
            logprobs[:, -response_length:],
            values_pred[:, -response_length:],
            attention_mask[:, -response_length:],
        )

        loss, stats = self.config.method.loss(
            logprobs=logprobs,
            values=values_pred,
            old_logprobs=old_logprobs,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            mask=mask,
        )
        self.approx_kl = stats["policy/approx_kl"]  # Update kl controller stats
        return loss, stats

    def post_epoch_callback(self):
        self.store.clear_history()
        self.orch.make_experience(
            self.config.method.num_rollouts, self.iter_count
        )  # Collect more rollouts for training

    def post_backward_callback(self):
        self.kl_ctl.update(self.approx_kl, n_steps=self.config.train.batch_size)

    def prepare_learning(self):
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        train_dataloader = self.store.create_loader(
            self.config.train.batch_size, shuffle=True
        )

        self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            train_dataloader, eval_dataloader
        )

        self.n_updates_per_batch = self.config.method.ppo_epochs
        self.total_steps = (
            self.config.train.epochs
            * self.n_updates_per_batch
            * len(self.train_dataloader)
        )
        self.total_steps = min(self.total_steps, self.config.train.total_steps)
