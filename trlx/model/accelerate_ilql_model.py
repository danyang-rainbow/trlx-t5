from typing import Iterable, Sequence, Union, cast

import torch
import torch.nn.functional as F


from trlx.model import register_model
from trlx.model.nn.ilql_models import ILQLConfig, CausalLMWithValueHeads
from trlx.data.ilql_types import ILQLBatch
from trlx.data.configs import TRLConfig
from trlx.utils import to_device

from .accelerate_base_model import AccelerateRLModel


@register_model
class AccelerateILQLModel(AccelerateRLModel):
    def __init__(
        self,
        config: TRLConfig,
        logit_mask=None,
        metric_fn=None,
        train_mode=True,
    ):
        super().__init__(config, train_mode)
        self.logit_mask = logit_mask
        self.metric_fn = metric_fn
        self.reward_fn = None

        if not isinstance(config.method, ILQLConfig):
            raise ValueError("config.method must be ILQLConfig")

        self.ilql: ILQLConfig = cast(ILQLConfig, config.method)

    def get_arch(self, config):
        return CausalLMWithValueHeads(
            config.model.model_path,
            ilql_config=config.method,
            num_layers_unfrozen=config.model.num_layers_unfrozen,
        )

    def tokenize(self, texts: Union[Sequence[str], Sequence[torch.LongTensor]]):
        if isinstance(texts[0], torch.LongTensor):
            return texts

        tokenized = self.tokenizer(
            [self.tokenizer.bos_token + x + self.tokenizer.eos_token for x in texts],
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = list(map(torch.as_tensor, tokenized.input_ids))
        return input_ids

    def post_backward_callback(self):
        if self.iter_count % self.config.method.steps_for_target_q_sync == 0:
            self.accelerator.unwrap_model(self.model).sync_target_q_heads()

    def loss(self, batch: ILQLBatch):
        batch = to_device(batch, self.accelerator.device)

        logits, qs, target_qs, vs, _ = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            actions_ixs=batch.actions_ixs,
            states_ixs=batch.states_ixs,
        )

        return self.ilql.loss((logits, (qs, target_qs, vs)), batch)

    def prepare_learning(self):
        train_dataloader = self.store.create_loader(self.config.train.batch_size)
        eval_dataloader = self.eval_pipeline.create_loader(self.config.train.batch_size)

        (
            self.model,
            self.opt,
            self.train_dataloader,
            self.eval_dataloader,
        ) = self.accelerator.prepare(
            self.model, self.opt, train_dataloader, eval_dataloader
        )

        self.n_updates_per_batch = 1
        self.total_steps = self.config.train.epochs * len(train_dataloader)
        self.total_steps = min(self.total_steps, self.config.train.total_steps)

        self.generate_kwargs = {
            "beta": self.config.method.betas[0],
            "max_length": self.max_length,
            "logit_mask": self.logit_mask,
            "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else 0,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else 0,
        }
