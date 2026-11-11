"""OpenRLHF dataset that retains per-pair reliability metadata."""

from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import Dataset

from openrlhf.datasets.reward_dataset import preprocess_data
from openrlhf.datasets.utils import zero_pad_sequences


class TRORewardDataset(Dataset):
    """DPO pair dataset with p_hat, KL endpoints, and lambda in each item.

    The prompt and answer fields remain ordinary chat messages.  No teacher
    score or p_star is consumed except for the explicitly named oracle method.
    """

    def __init__(self, dataset, tokenizer: Callable, max_length: int, strategy, input_template=None, is_dpo=True, num_processors=8, multiple_of=1):
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.multiple_of = multiple_of
        self.input_template = input_template
        self.prompt_key = getattr(strategy.args, "prompt_key", None)
        self.chosen_key = getattr(strategy.args, "chosen_key", "chosen")
        self.rejected_key = getattr(strategy.args, "rejected_key", "rejected")
        self.apply_chat_template = getattr(strategy.args, "apply_chat_template", False)
        if self.apply_chat_template:
            self.apply_chat_template = tokenizer.apply_chat_template
            template = getattr(strategy.args, "tokenizer_chat_template", None)
            if template:
                tokenizer.chat_template = template
        processed = dataset.map(self.process_data, remove_columns=dataset.column_names, num_proc=num_processors)
        processed = processed.filter(lambda x: x["prompt"] is not None)
        self.prompts = processed["prompt"]
        self.chosens = processed["chosen"]
        self.rejects = processed["reject"]
        self.extras = processed["extra"]

    def process_data(self, data):
        prompt, chosen, reject, prompt_id_len = preprocess_data(
            data,
            self.input_template,
            self.prompt_key,
            self.chosen_key,
            self.rejected_key,
            self.apply_chat_template,
            True,
        )
        prompt_token = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        actual_prompt_len = prompt_token["attention_mask"].int().sum().item()
        if actual_prompt_len >= self.max_length - 2:
            prompt = None
        extra = {
            "prompt_id_len": actual_prompt_len,
            "p_hat": float(data.get("p_hat", 0.5)),
            "lower": float(data.get("lower", data.get("p_hat", 0.5))),
            "upper": float(data.get("upper", data.get("p_hat", 0.5))),
            "lambda_i": float(data.get("lambda_i", 0.0)),
            "p_star": float(data.get("p_star", 0.5)),
            "n_votes": int(data.get("n_votes", 0)),
        }
        return {"prompt": prompt, "chosen": chosen, "reject": reject, "extra": extra}

    def __len__(self):
        return len(self.chosens)

    def __getitem__(self, idx):
        prompt, chosen, reject, extra = self.prompts[idx], self.chosens[idx], self.rejects[idx], self.extras[idx]
        chosen_text = (prompt + chosen).rstrip("\n")
        reject_text = (prompt + reject).rstrip("\n")
        if not chosen_text.endswith(self.tokenizer.eos_token):
            chosen_text += " " + self.tokenizer.eos_token
        if not reject_text.endswith(self.tokenizer.eos_token):
            reject_text += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(chosen_text, max_length=self.max_length, padding=False, truncation=True, return_tensors="pt", add_special_tokens=False)
        reject_token = self.tokenizer(reject_text, max_length=self.max_length, padding=False, truncation=True, return_tensors="pt", add_special_tokens=False)
        chosen_token["input_ids"][0, -1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0, -1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0, -1] = True
        reject_token["attention_mask"][0, -1] = True
        return chosen_token["input_ids"], chosen_token["attention_mask"], reject_token["input_ids"], reject_token["attention_mask"], extra

    def collate_fn(self, item_list):
        chosen_ids, chosen_masks, reject_ids, reject_masks, extras = zip(*item_list)
        chosen_ids = zero_pad_sequences(list(chosen_ids), side="right", value=self.tokenizer.pad_token_id)
        chosen_masks = zero_pad_sequences(list(chosen_masks), side="right")
        reject_ids = zero_pad_sequences(list(reject_ids), side="right", value=self.tokenizer.pad_token_id)
        reject_masks = zero_pad_sequences(list(reject_masks), side="right")
        return chosen_ids, chosen_masks, reject_ids, reject_masks, list(extras)

    def packing_collate_fn(self, item_list):
        # Keep the exact OpenRLHF packed representation while carrying metadata.
        extras = []
        chosen_ids, chosen_masks, chosen_lens = [], [], []
        rejected_ids, rejected_masks, rejected_lens = [], [], []
        index = 1
        for chosen_id, chosen_mask, reject_id, reject_mask, extra in item_list:
            chosen_ids.append(chosen_id.flatten())
            chosen_masks.append(torch.full_like(chosen_id.flatten(), index))
            chosen_lens.append(len(chosen_id.flatten()))
            rejected_ids.append(reject_id.flatten())
            rejected_masks.append(torch.full_like(reject_id.flatten(), index + len(item_list)))
            rejected_lens.append(len(reject_id.flatten()))
            extras.append(extra)
            index += 1
        packed_input_ids = torch.cat(chosen_ids + rejected_ids, dim=0).unsqueeze(0)
        packed_attention_masks = torch.cat(chosen_masks + rejected_masks, dim=0).unsqueeze(0)
        if self.multiple_of > 1 and packed_input_ids.numel() % self.multiple_of != 0:
            padding_len = self.multiple_of - (packed_input_ids.numel() % self.multiple_of)
            packed_input_ids = torch.nn.functional.pad(packed_input_ids, (0, padding_len), value=self.tokenizer.pad_token_id)
            packed_attention_masks = torch.nn.functional.pad(packed_attention_masks, (0, padding_len), value=0)
        return packed_input_ids, packed_attention_masks, chosen_lens + rejected_lens, extras
