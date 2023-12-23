from typing import List
from transformers import AutoTokenizer


def exact_match(output: str, reference: str) -> int:
    return int(output == reference)


def levenshtein_distance(output: str, reference: str) -> int:
    matrix = [
        [i + j for j in range(len(reference) + 1)] for i in range(len(output) + 1)
    ]
    for i in range(1, len(output) + 1):
        for j in range(1, len(reference) + 1):
            if output[i - 1] == reference[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d
            )
    return matrix[len(output)][len(reference)]


def edit_sim(output: str, reference: str) -> float:
    return 1 - levenshtein_distance(output, reference) / max(
        len(output), len(reference)
    )


def ES(output: str, reference: str):
    return edit_sim(output=output, reference=reference)


def EM(output: str, reference: str):
    return exact_match(output=output, reference=reference)


if __name__ == "__main__":
    output = "public static void main(String[] args) {}"
    reference = "public static void main(String[] args) {"
    print(EM(output, reference))
    print(ES(output, reference))
