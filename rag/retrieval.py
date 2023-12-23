import json
import time
from loguru import logger


class Retrieval:
    def __init__(
        self,
        retrieval="random",
        snippets_path="./code-snippets.json",
        prompt_version: str = "v3",
        model_path: str = None,
    ):
        self.snippets_path = snippets_path
        self.code_snippets = self.get_code_snippets(self.snippets_path)
        self.is_filter_path = is_filter_path
        self.is_filter_gt = is_filter_gt
        self.prompt_version = prompt_version
        self.model_path = model_path

        if "starcoder" in self.model_path:
            self.FIM_PRE = "<fim_prefix>"
            self.FIM_SUF = "<fim_suffix>"
            self.FIM_MID = "<fim_middle>"
        else:
            self.FIM_PRE = " <PRE>"
            self.FIM_SUF = " <SUF>"
            self.FIM_MID = " <MID>"

        logger.info(f"Retrieval: {retrieval}")
        logger.info(f"total snippets: {len(self.code_snippets)}")
        logger.info(f"model_path: {self.model_path}")

        if retrieval == "random":
            from .random_retrieval import RandomRetrieval

            self.runner = RandomRetrieval(self.code_snippets)

        elif retrieval == "bm25":
            from .bm25_retrieval import BM25Retrieval

            self.runner = BM25Retrieval(self.code_snippets, self.model_path)

        elif retrieval == "jaccard":
            from .jaccard_retrieval import JaccardRetrieval

            self.runner = JaccardRetrieval(self.code_snippets, self.model_path)

        elif retrieval == "reacc":
            from .prompt_retrieval import Retrieval

            prompt_version = self.prompt_version
            self.runner = Retrieval(
                dataset_path=f"./rag/npz/snippets-gte-large-{self.prompt_version}-avg",
                model_name="./models/embedding/gte-large",
                max_seq_length=512,
                mode="avg",
                prompt_version=self.prompt_version,
            )

        elif retrieval == "codellama-13b-chat-avg":
            from .prompt_retrieval import Retrieval

            prompt_version = self.prompt_version
            self.runner = Retrieval(
                dataset_path=f"./rag/npz/snippets-kwai-codellama-13b-chat-{self.prompt_version}-avg",
                model_name="./models/CodeLlama-13b-Instruct-hf",
                max_seq_length=512,
                mode="avg",
                prompt_version=self.prompt_version,
            )

        elif retrieval == "codellama-13b-chat-generate":
            from .prompt_retrieval import Retrieval

            prompt_version = self.prompt_version
            self.runner = Retrieval(
                dataset_path=f"./rag/npz/snippets-kwai-codellama-13b-chat-{self.prompt_version}-generate",
                model_name="./models/CodeLlama-13b-Instruct-hf",
                max_seq_length=512,
                mode="generate",
                prompt_version=self.prompt_version,
            )

        elif retrieval == "starcoder-avg":
            from .prompt_retrieval import Retrieval

            prompt_version = self.prompt_version
            self.runner = Retrieval(
                dataset_path=f"./rag/npz/snippets-starcoder-{self.prompt_version}-avg",
                model_name="./model/starcoderbase",
                max_seq_length=512,
                mode="avg",
                prompt_version=self.prompt_version,
            )

        elif retrieval == "starcoder-generate":
            from .prompt_retrieval import Retrieval

            prompt_version = self.prompt_version
            self.runner = Retrieval(
                dataset_path=f"./rag/npz/snippets-starcoder-{self.prompt_version}-generate",
                model_name="./model/starcoderbase",
                max_seq_length=512,
                mode="generate",
                prompt_version=self.prompt_version,
            )

        elif retrieval == "bandit":
            from .bandit_retrieval import BanditRetreval

            self.runner = BanditRetreval(model_dir=self.model_path)

        else:
            raise NotImplementedError

    def release_model(self):
        self.runner.release_model()
        time.sleep(5)

    def get_code_snippets(self, snippets_path):
        with open(snippets_path, "r") as f:
            snippets = json.load(f)
        return snippets

    def query_top_k(self, code, best_of=50):
        top_k = self.runner.query_top_k(top_k=best_of, code=code)

        return top_k

    def get_query_code(self, prefix: str, suffix: str, window=30) -> str:
        code_context = []
        line_count = 0

        if prefix.strip():
            lines_before = prefix.split("\n")
            start_index = max(0, len(lines_before) - window // 2)
            for line in lines_before[start_index:]:
                code_context.append(line)
                line_count += 1

        if suffix.strip():
            lines_after = suffix.split("\n")
            for i, line in enumerate(lines_after):
                if i >= window - line_count:
                    break
                code_context.append(line)

        return "\n".join(code_context)

    def get_bandit(self, task_id):
        top_k = self.runner.query_top_k(task_id=task_id)
        return top_k

    def get_query_code_with_fim(self, prefix: str, suffix: str, window=30) -> str:
        code_before = []
        code_after = []
        line_count = 0

        if prefix.strip():
            lines_before = prefix.split("\n")
            start_index = max(0, len(lines_before) - window // 2)
            for line in lines_before[start_index:]:
                code_before.append(line)
                line_count += 1

        if suffix.strip():
            lines_after = suffix.split("\n")
            for i, line in enumerate(lines_after):
                if i >= window - line_count:
                    break
                code_after.append(line)

        _prefix = "\n".join(code_before) + "\n"
        _suffix = "\n" + "\n".join(code_after)
        return self.FIM_PRE + _prefix + self.FIM_SUF + _suffix + self.FIM_MID
