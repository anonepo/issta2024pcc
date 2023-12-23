from tqdm import tqdm
import json
import datasets
from datasets import Dataset
from transformers import AutoTokenizer
import heapq


class JaccardRetrieval:
    def __init__(self, code_snippets, model_name):
        self.max_seq_length = 512
        self.code_snippets = code_snippets
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dataset_code = Dataset.from_list(self.code_snippets)

        dataset_code_token = dataset_code.map(
            self.tokenize_function,
            batched=True,
            num_proc=8,
            # remove_columns=['text_column_name'],
        )

        code_snippets_tokens = dataset_code_token["input_ids"]
        self.code_snippets_tokens_set = [set(item) for item in code_snippets_tokens]

    def sim_jaccard(self, s1, s2):
        """jaccard相似度"""
        # s1, s2 = set(s1), set(s2)
        ret1 = s1.intersection(s2)
        ret2 = s1.union(s2)
        sim = 1.0 * len(ret1) / len(ret2)
        return sim

    def tokenize_function(self, examples):
        return self.tokenizer(
            examples["content"],
            padding=False,
            truncation=True,
            max_length=self.max_seq_length,
            return_special_tokens_mask=False,
        )

    def query_top_k(self, top_k, code):
        query_set = set(self.tokenizer(code)["input_ids"])
        result = []

        for idx, snippet in enumerate(self.code_snippets_tokens_set):
            distance = self.sim_jaccard(query_set, snippet)

            if len(result) < top_k or distance > min(item[0] for item in result):
                # Note that we store the positive distance here, so we can use min-heap properties
                res = {
                    "code": self.code_snippets[idx]["content"],
                    "language": self.code_snippets[idx]["language"],
                    "path": self.code_snippets[idx]["path"],
                    "distance": distance,
                }

                if len(result) < top_k:
                    heapq.heappush(result, (distance, idx, res))
                else:
                    heapq.heappop(result)
                    heapq.heappush(result, (distance, idx, res))

        result = [item[2] for item in heapq.nlargest(top_k, result, key=lambda x: x[0])]
        result = sorted(result, key=lambda x: x["distance"], reverse=True)
        return result


if __name__ == "__main__":
    retrieval = JaccardRetrieval()
    code = """
    def test():
        print("test")
    """

    top_k = retrieval.query_top_k(10, code)
    print(top_k)
