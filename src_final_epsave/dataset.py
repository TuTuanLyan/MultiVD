import torch
from torch.utils.data import Dataset


class CodeDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        tokenizer,
        max_length: int,
        truncation_strategy: str = "head_middle_tail",
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy

    def _truncate(self, token_ids: list[int], budget: int) -> list[int]:
        if len(token_ids) <= budget:
            return token_ids
        if self.truncation_strategy == "head":
            return token_ids[:budget]
        if self.truncation_strategy != "head_middle_tail":
            raise ValueError(f"unsupported truncation strategy: {self.truncation_strategy}")

        # Long vulnerability patches can occur far from either end. Retain three
        # deterministic windows so paired samples do not become identical prefixes.
        head_size = budget // 3 + budget % 3
        middle_size = budget // 3
        tail_size = budget // 3
        middle_start = (len(token_ids) - middle_size) // 2
        return (
            token_ids[:head_size]
            + token_ids[middle_start : middle_start + middle_size]
            + token_ids[-tail_size:]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        encoded = self.tokenizer(
            record["code"],
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            verbose=False,
        )
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        budget = self.max_length - special_tokens
        if budget < 3:
            raise ValueError("max_length is too small for model special tokens and three windows")
        token_ids = self._truncate(encoded["input_ids"], budget)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(token_ids)
        attention_mask = [1] * len(input_ids)
        padding = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding
        attention_mask += [0] * padding

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        item["labels"] = torch.tensor(int(record["label"]), dtype=torch.long)
        item["cwe_class"] = torch.tensor(int(record.get("cwe_class", -100)), dtype=torch.long)
        item["index"] = torch.tensor(index, dtype=torch.long)
        return item
