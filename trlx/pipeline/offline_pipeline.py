from typing import Iterable

import torch
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer

from trlx.data.ilql_types import ILQLBatch, ILQLElement
from trlx.pipeline import BasePipeline, BaseRolloutStore, register_datapipeline


@dataclass
class DataCollatorForRLUL2:
    tokenizer:AutoTokenizer = None

    def __call__(self, features):
        new_features = {
            "prompts":[one["prompts"] for one in features],
            "response_gt":[one["response_gt"] for one in features],
        }
        features = new_features
        batch = self.tokenizer(features["prompts"], padding="max_length", truncation=True, max_length=512, add_special_tokens=False, return_tensors="pt")
        
        batch["response_gt"] = features["response_gt"]
        return batch

@register_datapipeline
class PromptPipeline(BasePipeline):
    """
    Tokenizes texts, and then pads them into batches
    """

    def __init__(self, prompts, response_gt,tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.response_gt = response_gt

    def __getitem__(self, ix: int):
        return {
            "prompts": self.prompts[ix],
            "response_gt": self.response_gt[ix]
        }

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        collate_fn = DataCollatorForRLUL2(self.tokenizer)
        return DataLoader(
            self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle
        )


class ILQLRolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training ILQL
    """

    def __init__(
        self, input_ids, attention_mask, rewards, states_ixs, actions_ixs, dones
    ):
        super().__init__()

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.rewards = rewards
        self.states_ixs = states_ixs
        self.actions_ixs = actions_ixs
        self.dones = dones

    def __getitem__(self, ix: int) -> ILQLElement:
        return ILQLElement(
            self.input_ids[ix],
            self.attention_mask[ix],
            self.rewards[ix],
            self.states_ixs[ix],
            self.actions_ixs[ix],
            self.dones[ix],
        )

    def __len__(self) -> int:
        return len(self.input_ids)

    def create_loader(self, batch_size: int):
        def collate_fn(elems: Iterable[ILQLElement]):
            return ILQLBatch(
                pad_sequence(
                    [x.input_ids for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.attention_mask for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.rewards for x in elems], batch_first=True, padding_value=0.0
                ),
                pad_sequence(
                    [x.states_ixs for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.actions_ixs for x in elems], batch_first=True, padding_value=0
                ),
                pad_sequence(
                    [x.dones for x in elems], batch_first=True, padding_value=0
                ),
            )

        return DataLoader(
            self, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
