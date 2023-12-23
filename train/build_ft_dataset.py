import json
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process training data for Code Llama model."
    )
    parser.add_argument(
        "--model_dir", required=True, help="Directory where the model is stored"
    )
    parser.add_argument(
        "--data_dir", required=True, help="Directory where the training data is located"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where the formatted training data will be saved",
    )
    return parser.parse_args()


def truncate_and_format(
    data,
    tokenizer,
    output_token_limit=128,
    total_token_limit=4096,
    prefix_ratio=0.85,
    prefix_tag="<fim_prefix>",
    middle_tag="<fim_middle>",
    suffix_tag="<fim_suffix>",
):
    formatted_data = {"type": "text2text", "instances": []}

    for item in tqdm(data, desc="format training datasets..."):
        # Tokenize prefix and suffix
        tokens_prefix = tokenizer.encode(item["prefix"], add_special_tokens=False)
        tokens_suffix = tokenizer.encode(item["suffix"], add_special_tokens=False)
        tokens_output = tokenizer.encode(item["reference"], add_special_tokens=False)

        # Ensure output is within output_token_limit tokens
        if len(tokens_output) > output_token_limit:
            continue

        # Calculate token limits for prefix and suffix
        total_length_limit = total_token_limit - output_token_limit
        prefix_limit = int(total_length_limit * prefix_ratio)
        suffix_limit = total_length_limit - prefix_limit

        # Truncate prefix and suffix
        tokens_prefix = tokens_prefix[-prefix_limit:]
        tokens_suffix = tokens_suffix[:suffix_limit]

        # Combine and format input
        input_text = (
            prefix_tag
            + tokenizer.decode(tokens_prefix)
            + suffix_tag
            + tokenizer.decode(tokens_suffix)
            + middle_tag
        )

        output_text = tokenizer.decode(tokens_output) + tokenizer.eos_token

        formatted_instance = {
            "input": input_text,
            "output": output_text,
        }
        formatted_data["instances"].append(formatted_instance)

    return formatted_data


def main(args):
    # load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)

    # load the data.
    with open(args.data_dir, "r") as file:
        train_data = json.load(file)

    # format the data to text2text.
    formatted_train_data = truncate_and_format(train_data, tokenizer)

    with open(args.output_dir, "w") as file:
        json.dump(formatted_train_data, file, indent=4)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
