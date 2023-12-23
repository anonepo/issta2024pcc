from transformers import (
    AutoTokenizer,
    AutoModel,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
import torch
from sklearn.preprocessing import normalize
import numpy as np
import time
from torch import Tensor
from loguru import logger
import faiss
import gc
from loguru import logger
from .prompt_wrapper import PromptWrapper
from .prompt_embedding import STOP_SIGN, SPECIAL
from transformers import set_seed

set_seed(42)


class Retrieval:
    def __init__(
        self,
        dataset_path: str,
        model_name: str,
        max_seq_length=512,
        mode="avg",
        prompt_version="v1",
        model=None,
    ):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.prompt_version = prompt_version
        logger.info(f"dataset_path: {dataset_path}")
        logger.info(f"model_name: {model_name}")
        logger.info(f"max_seq_length: {max_seq_length}")
        logger.info(f"mode: {mode}")
        logger.info(f"Prompt Version: {prompt_version}")

        time1 = time.time()
        tmp = np.load(self.dataset_path + ".npz", allow_pickle=True)
        self.content = tmp["content"].tolist()
        self.path = tmp["path"].tolist()
        self.language = tmp["language"].tolist()
        if self.mode == "generate":
            self.block = tmp["block"].tolist()

        embs = tmp["embs"]  # ,dtype=np.float16)
        tmp.close()
        self.dim = len(embs[0])  # 4096.
        logger.info(f"faiss embedding dim: {self.dim}")
        logger.info(f"total code snippets: {len(embs)}")

        self.cpu_index = faiss.IndexFlatL2(self.dim)  # 构建索引index
        self.cpu_index.add(embs)
        time2 = time.time()

        logger.info("read data:{:.5f}".format(time2 - time1))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if model:
            self.model = model
        else:
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

        self.model.tie_weights()
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.prompt_wrapper = PromptWrapper(prompt_version=prompt_version)
        logger.info(f"Prompt is {self.prompt_wrapper.prompt()}")

    def release_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def average_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def convert_to_similar_code_format(self, dis, ind):
        results = []
        for d, i in zip(dis, ind):
            if self.mode == "generate":
                code = self.block[i]
            else:
                code = self.content[i]
            results.append(
                {
                    "code": code,
                    "language": self.language[i],
                    "path": self.path[i],
                    "distance": float(d),
                }
            )
        results = sorted(results, key=lambda x: x["distance"])
        return results

    def query_top_k(self, code, top_k=50):
        input = self.tokenizer(
            self.prompt_wrapper.wrapper(code),
            truncation=True,
            max_length=512,
            # return_tensors="pt",
        )
        with torch.no_grad():
            if self.mode == "generate":
                embeddings = None  ############here for generation, prefix+sufix+ <MID>
                err = False
                input_ids = torch.tensor([input["input_ids"]])
                skip = [len(input_ids[0])]
                for idx in range(21):  # max length for 20
                    outputs = self.model(input_ids=input_ids, output_hidden_states=True)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
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
                    err = True
                good_idx = [
                    idx
                    for idx, val in enumerate(input_ids[0][skip[-1] : -1])
                    if val.item() not in SPECIAL
                    and self.tokenizer.decode(val.item()).strip()
                ]  # exclude special tokens and space
                if len(good_idx) == 0:
                    err = True

                embeddings = embeddings[good_idx]
                embeddings = embeddings.mean(dim=0, keepdim=True)  ######### average

                if err:
                    print("error")
                    embeddings = torch.zeros((1, self.dim), dtype=torch.float32)

                embeddings = embeddings.float().cpu().numpy()
                embeddings = np.asarray(embeddings, dtype=np.float32)
                embeddings = normalize(embeddings)

            else:
                outputs = self.model(
                    input_ids=torch.tensor([input["input_ids"]]),
                    attention_mask=torch.tensor([input["attention_mask"]]),
                    output_hidden_states=True,
                    return_dict=True,
                )
                if self.mode == "avg":
                    embeddings = (
                        self.average_pool(
                            outputs.last_hidden_state,
                            torch.tensor([input["attention_mask"]]),
                        )
                        .float()
                        .cpu()
                        .numpy()
                    )
                elif self.mode == "last":
                    embeddings = (
                        outputs.last_hidden_state[:, -1, :].float().cpu().numpy()
                    )
                else:
                    raise NotImplementedError

            embeddings = np.asarray(embeddings, dtype=np.float32)
            embeddings = normalize(embeddings)
            dis, ind = self.cpu_index.search(embeddings, top_k)
            results = self.convert_to_similar_code_format(dis[0], ind[0])
            del outputs
        del input
        return results
