import random


class RandomRetrieval:
    def __init__(self, code_snippets):
        self.code_snippets = code_snippets

    def query_top_k(self, top_k=50, code=None):
        snippets = random.sample(self.code_snippets, top_k)
        result = [
            {
                "code": snippet["content"],
                "language": snippet["language"],
                "path": snippet["path"],
                "distance": 0,
            }
            for snippet in snippets
        ]
        return result
