import json
import sys

sys.setrecursionlimit(3000)
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_from_disk
from tqdm import tqdm
import argparse


def split_sample_with_fim(content, window=30):
    snippets = content.split("\n")
    snippet_list = []

    for idx in range(1, len(snippets) - 1):
        prefix_lines = snippets[:idx]
        current_line = snippets[idx]
        suffix_lines = snippets[idx + 1 :]

        if len(current_line.strip()) < 3:
            continue

        line_count = 0
        prefix_context = []
        suffix_context = []

        start_index = max(0, len(prefix_lines) - window // 2)
        for line in prefix_lines[start_index:]:
            prefix_context.append(line)
            line_count += 1

        for i, line in enumerate(suffix_lines):
            if i >= window - line_count:
                break
            suffix_context.append(line)

        block_lines = prefix_context + [current_line] + suffix_context

        _prefix = "\n".join(prefix_context) + "\n"
        _suffix = "\n" + "\n".join(suffix_context)

        if model_type == "starcoder":
            combined_snippet = (
                "<fim_prefix>" + _prefix + "<fim_suffix>" + _suffix + "<fim_middle>"
            )
        else:
            combined_snippet = " <PRE>" + _prefix + " <SUF>" + _suffix + " <MID>"

        snippet_list.append(
            {
                "content": combined_snippet,
                "start_line": idx + 1,
                "type": "snippet_with_fim",
                "block": "\n".join(block_lines),
            }
        )

    return snippet_list


def split_sample(content, window_length=30):
    snippets = content.split("\n")
    snippet_list = []
    # 没有overlap
    for l in range(0, len(snippets), window_length):
        lines = snippets[l : l + window_length]
        # 小于10行的代码不入库
        if len(lines) < 10:
            break
        snippet = "\n".join(lines)
        snippet = snippet + "\n"
        snippet_list.append(
            {
                "content": snippet,
                "start_line": l + 1,
                "type": "snippets",
                "block": snippet,
            }
        )
    return snippet_list


def process_dataset(data_dir):
    ds = load_from_disk(data_dir)

    samples = []
    for content, language, path in zip(ds["content"], ds["language"], ds["path"]):
        if path not in train:
            continue

        sample = {"content": content, "language": language, "path": path}
        samples.append(sample)

    print(f"Processing {len(samples)} code files...")

    all_snippets = []
    for sample in tqdm(samples, total=len(samples), desc="Construct code corpus"):
        if use_fim:
            snippets = split_sample_with_fim(sample["content"])
        else:
            snippets = split_sample(sample["content"])
        for snippet in snippets:
            all_snippets.append(
                {
                    "content": snippet["content"],
                    "language": sample["language"],
                    "path": sample["path"],
                    "start_line": snippet["start_line"],
                    "block": snippet["block"],
                }
            )

    print(f"totol {len(all_snippets)} snippets")

    with open(output_file, "w") as json_file:
        json.dump(all_snippets, json_file, indent=4)

    print(f"data save to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="../dataset/java_pub.hf")
    parser.add_argument("--output_file", type=str, default="./code-snippets.json")
    parser.add_argument("--data_split", type=str, default="../dataset/data_split.json")
    parser.add_argument("--fim", type=bool, default=False)
    parser.add_argument("--model_type", default="code_llama")

    return parser.parse_args()


if __name__ == "__main__":
    args = parser.parse_args()

    dataset_dir = args.data_dir
    output_file = args.output_file
    use_fim = args.fim
    model_type = args.model_type

    with open(args.data_split, "r") as f:
        data_split = json.load(f)

    train = data_split["train"]
    process_dataset(dataset_dir)
