from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForCausalLM,
)
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import normalize
import numpy as np
from datasets import Dataset
import json
import argparse
from torch import Tensor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from loguru import logger
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
)
from loguru import logger
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.prompt_wrapper import PromptWrapper

STOP_SIGN = 13
SPECIAL = [
    1,
    2,
    32007,
    32008,
    32009,
    32010,
]  #'<s>', '</s>', '▁<PRE>', '▁<MID>', '▁<SUF>', '▁<EOT>'


@dataclass
class MyDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = (512,)
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        content = []
        path = []
        language = []
        block = []
        for one in features:
            content.append(one.pop("content"))
            path.append(one.pop("path"))
            language.append(one.pop("language"))
            block.append(one.pop("block"))

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        batch["content"] = content
        batch["path"] = path
        batch["language"] = language
        batch["block"] = block
        return batch


class Embedding:
    def __init__(
        self,
        dataset_path: str,
        model_name: str,
        max_seq_length,
        batch_size,
        save,
        mode,
        prompt_version="v1",
    ):
        logger.info(f"Prompt version {prompt_version}")
        logger.info(f"mode {mode}")
        logger.info(f"model {model_name}")

        self.dataset_path = dataset_path
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.save = save
        self.mode = mode
        self.prompt_version = prompt_version
        self.prompt_wrapper = PromptWrapper(prompt_version=self.prompt_version)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.mode == "generate":
            if "starcoder" in self.model_name:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            else:
                self.model = LlamaForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
        else:
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )

        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["content"],
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_special_tokens_mask=False,
        )

    def average_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embedding(self):
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            code_snippets = json.load(f)

        dataset_code = Dataset.from_list(code_snippets)

        dataset_code = dataset_code.map(self.prompt_wrapper.apply_prompt, batched=False)

        dataset_code_token = dataset_code.map(
            self.tokenize_function,
            batched=True,
            num_proc=1,
            # remove_columns=[text_column_name],
        )
        logger.info(len(dataset_code_token))

        content = []
        path = []
        language = []
        block = []
        embs = []

        batchify_fn = MyDataCollatorWithPadding(
            tokenizer=self.tokenizer, max_length=self.max_seq_length
        )

        if self.mode == "generate":
            self.batch_size = 1

        data_loader = DataLoader(
            dataset_code_token,
            shuffle=False,
            collate_fn=batchify_fn,
            batch_size=self.batch_size,
        )

        progress_bar = tqdm(range(len(data_loader)))
        with torch.no_grad():
            for step, batch in enumerate(data_loader):
                if self.mode == "generate":
                    embeddings = (
                        None  ############here for generation, prefix+sufix+ <MID>
                    )
                    skip = [len(batch["input_ids"][0])]
                    input_ids = batch["input_ids"].cuda()
                    for idx in range(21):  # max length for 20
                        outputs = self.model(
                            input_ids=input_ids, output_hidden_states=True
                        )
                        next_token_logits = outputs.logits[:, -1, :]
                        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(
                            -1
                        )
                        input_ids = torch.cat([input_ids, next_token], dim=-1)
                        if next_token[0].item() in [
                            STOP_SIGN,
                            self.tokenizer.eos_token_id,
                            32010,
                        ]:  # only consider one line,exclude <EOT>
                            if len(skip) == idx + 1:
                                skip.append(len(input_ids[0]))
                            else:
                                embeddings = outputs.hidden_states[-1][0][
                                    skip[-1] :
                                ]  # -1 is the last hidden states, but only consider the generated texts (len(batch['input_ids][0] to last!))
                                break
                    if embeddings == None:
                        embeddings = outputs.hidden_states[-1][0][skip[-1] :]
                    if len(embeddings) < 1:
                        continue
                    good_idx = [
                        idx
                        for idx, val in enumerate(input_ids[0][skip[-1] : -1])
                        if val.item() not in SPECIAL
                        and self.tokenizer.decode(val.item()).strip()
                    ]  # exclude special tokens and space
                    if len(good_idx) == 0:
                        continue
                    embeddings = embeddings[good_idx]
                    embeddings = embeddings.mean(dim=0, keepdim=True)  #########average
                    if torch.isnan(embeddings).any():
                        continue
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    if self.mode == "avg":
                        embeddings = self.average_pool(
                            outputs.last_hidden_state,
                            batch["attention_mask"],
                        ).float()
                    elif self.mode == "last":
                        embeddings = outputs.last_hidden_state[:, -1, :].float()
                    else:
                        raise NotImplementedError

                de_prompted_content = [
                    self.prompt_wrapper.de_wrapper(c) for c in batch["content"]
                ]
                content.extend(de_prompted_content)
                path.extend(batch["path"])
                language.extend(batch["language"])
                block.extend(batch["block"])
                embs.extend(embeddings.float().cpu().numpy().tolist())
                progress_bar.update(1)

            embs = normalize(embs)

            np.savez(
                f"{self.save}-{self.prompt_version}-{self.mode}",
                embs=embs.astype(np.float32),
                content=np.array(content),
                path=np.array(path),
                language=np.array(language),
                block=np.array(block),
            )


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="./rag/code-snippets.json",
        type=str,
        required=False,
        help="dataset name",
    )
    parser.add_argument(
        "--model_name",
        default="./models/embedding/CodeLlama-13b-Instruct-hf",
        type=str,
        required=False,
        help="tokenizer name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=1024,
        type=int,
    )
    parser.add_argument("--batch_size", default=8, type=int, help="batch size.")
    parser.add_argument(
        "--save",
        default="./npz/snippets-codellama-13b-chat-v1-last",
        type=str,
        required=False,
        help="save name",
    )
    parser.add_argument("--mode", default="avg", type=str)
    parser.add_argument("--prompt_version", default="v1", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseargs()
    embd = Embedding(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        save=args.save,
        mode=args.mode,
        prompt_version=args.prompt_version,
    )
    embd.embedding()