import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextualbandits.online import LinUCB
import numpy as np
import json
import torch
import gc
from transformers import AutoTokenizer
from collections import Counter
import random

random.seed(42)


class BanditRetreval:
    def __init__(self, model_dir):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.arm_results = [
            "./eval/results/code_llama_generate.json",
            "./eval/results/code_llama_generate_v5.json",
            "./eval/results/code_llama_avg_v3.json",
        ]

        nchoices = len(self.arm_results)
        print(f"Using {nchoices} arms")
        self.nchoices = nchoices

        self.bandit = LinUCB(nchoices=nchoices, alpha=0.1)
        self.train()

        print(f"Bandit Contextual {self.bandit.__class__}")

        self.arm_retrievals = [
            self.load_retrieval_data(arm) for arm in self.arm_results
        ]

        self.chosen_arms = []

    def load_retrieval_data(self, path):
        with open(path, "r") as file:
            data = json.load(file)
        retrieval_data = {
            item["task_id"]: {
                "query_code": item.get("query_code", None),
                "retrieval": item.get("retrieval", None),
                "output": [item["output"]],
            }
            for item in data
        }
        return retrieval_data

    def query_top_k(self, task_id=None):
        ret_data = [arm.get(task_id, {}) for arm in self.arm_retrievals]
        ret_output = [data.get("output", "") for data in ret_data]
        ret_sim = [rag[0].get("distance", 0) for rag in ret_rag]

        # gte jaccard sim
        jaccard_sims = []
        for ret_data in ret_data:
            query_code = ret_data.get("query_code", "")
            retrieved_code = ret_data.get("retrieval", [{}])[0].get("code", "")
            if query_code and retrieved_code:
                query_code_tokens = set(self.tokenizer.tokenize(query_code))
                retrieved_code_tokens = set(self.tokenizer.tokenize(retrieved_code))
                jaccard_score = self.calculate_jaccard_similarity(
                    query_code_tokens, retrieved_code_tokens
                )
                jaccard_sims.append(jaccard_score)
            else:
                jaccard_sims.append(0)

        feature = np.array(ret_sim + jaccard_sims)
        feature = np.array([feature])

        chosen_arm = self.bandit.predict(feature)[0]

        chosen_ret_rag = ret_rag[chosen_arm]
        chosen_ret_output = ret_output[chosen_arm]
        return chosen_ret_rag, chosen_ret_output

    def release_model(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def calculate_jaccard_similarity(self, set1, set2):
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union) if len(union) != 0 else 0

    def get_results_and_scores(self, path, tokenizer):
        with open(path, "r") as f:
            results = json.load(f)

        outputs = []
        max_similarity = []
        jaccard_similarity = []

        for result in results:
            outputs.append(result["output"])
            rags = result["retrieval"]

            query_code = result.get("query_code", "")
            retrieved_code = rags[0]["code"] if rags and len(rags) > 0 else ""

            if rags is not None and query_code and retrieved_code:
                max_sim = rags[0]["distance"]
                query_code_tokens = set(tokenizer.tokenize(query_code))
                retrieved_code_tokens = set(tokenizer.tokenize(retrieved_code))
                jaccard_score = self.calculate_jaccard_similarity(
                    query_code_tokens, retrieved_code_tokens
                )
                jaccard_similarity.append(jaccard_score)
            else:
                max_sim = 0
                jaccard_similarity.append(0)

            max_similarity.append(max_sim)

        return outputs, max_similarity, jaccard_similarity

    def train(self):
        global debug
        val_paths = "../dataset/val.json"

        val_arm_results = [
            "./eval/valid_results/code_llama_generate.json",
            "./eval/valid_results/code_llama_generate_v5.json",
            "./eval/valid_results/code_llama_avg_v3.json",
        ]
        val_arm_results = self.arm_results

        with open(val_paths, "r") as f:
            valset = json.load(f)

        references = [val["reference"] for val in valset]

        arm_data = [
            self.get_results_and_scores(arm_result, self.tokenizer)
            for arm_result in val_arm_results
        ]

        features = []
        rewards = []
        chosen_arms = []
        for idx, val in enumerate(valset):
            feature = [data[1][idx] for data in arm_data] + [
                data[2][idx] for data in arm_data
            ]

            features.append(feature)

            correct_arms = []

            for arm_idx, data in enumerate(arm_data):
                if data[0][idx] == references[idx]:
                    correct_arms.append(arm_idx)

            if correct_arms:
                chosen_arm = random.choice(correct_arms)
            else:
                chosen_arm = random.randint(0, self.nchoices - 1)

            chosen_arms.append(chosen_arm)

            output = arm_data[chosen_arm][0][idx]
            reward = 1 if output == references[idx] else 0
            rewards.append(reward)

        print(Counter(chosen_arms))

        features = np.array(features)
        print(features[0])

        self.bandit.fit(features, np.array(chosen_arms), np.array(rewards))
