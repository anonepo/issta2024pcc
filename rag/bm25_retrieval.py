from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer


class BM25Retrieval:
    def __init__(self, code_snippets, model_name):
        self.code_snippets = code_snippets
        print(f"total snippets: {len(self.code_snippets)}")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        tokenized_texts = [
            self.tokenizer.tokenize(snippet["content"])
            for snippet in self.code_snippets
        ]
        self.bm25 = BM25Okapi(tokenized_texts)

    def query_top_k(self, top_k, code):
        tokenized_query = self.tokenizer.tokenize(code)

        doc_scores = self.bm25.get_scores(tokenized_query)

        top_indexes = sorted(
            range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True
        )[:top_k]

        result = [
            {
                "code": self.code_snippets[i]["content"],
                "language": self.code_snippets[i]["language"],
                "path": self.code_snippets[i]["path"],
                "distance": doc_scores[i],
            }
            for i in top_indexes
        ]

        return result
