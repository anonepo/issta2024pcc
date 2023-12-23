import asyncio
import json
import os
import pandas as pd
import sys
import time
import traceback

from argparse import ArgumentParser
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer


from interface.codellama import CodeLLamaInterface
from interface.starcoder import StarCoderInterface
from server.text_generation import TextGenerationServer, TextGenerationClient
from utils.metrics import EM, ES
from rag.retrieval import Retrieval
from tqdm import tqdm
from prettytable import PrettyTable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument(
        "--model_name", choices=["code_llama", "starcoder"], default="code_llama"
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument(
        "--code_snippets_path",
        type=str,
    )
    parser.add_argument(
        "--testset_path",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        type=str,
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=-1,
        help="Number of sampled testsets (-1 for all the testcases)",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "rag"],
        default="normal",
    )
    parser.add_argument(
        "--retrieval",
        choices=[
            "random",
            "bm25",
            "jaccard",
            "reacc",
            "bandit",
            "codellama-13b-chat-avg",
            "codellama-13b-chat-generate",
            "starcoder-avg",
            "starcoder-generate",
        ],
        default=None,
    )

    parser.add_argument("--release_model", type=bool, default=False)
    parser.add_argument("--prompt_version", type=str, default=None)
    parser.add_argument("--fim_prompt", type=bool, default=False)
    parser.add_argument("--oracle", type=bool, default=False)
    parser.add_argument("--use_local_rag", type=bool, default=False)
    parser.add_argument("--total_budget", type=int, default=4096)
    parser.add_argument("--max_rag_num", type=int, default=1)
    return parser.parse_args()


def run_eval_pipeline(args: ArgumentParser) -> int:
    """
    Run the Kwaipilot evaluation pipeline
    Args:
        run: wandb run object
        args: parsed arguments
    """

    model_path = Path(args.model_path)
    if not model_path.exists() or not model_path.is_dir():
        logger.error(f"Invalid model {model_path}")
        return -1

    try:
        testsets = json.load(open(args.testset_path, "r"))
        testsets = testsets[: args.sample_num] if args.sample_num > 0 else testsets
        # testsets = [testset for testset in testsets if testset['type'] == 'lines']
        logger.info(f"Loaded testset with {len(testsets)} cases")

        # Preprocess testset
        inputs = []
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        if args.model_name == "starcoder":
            stop_sequences = ["<|endoftext|>"]
        else:
            stop_sequences = [
                " <EOT>",
                "<EOT>",
                tokenizer.eos_token,
                tokenizer.eot_token,
            ]

        results = []

        if args.mode == "rag":
            # 本地存在则直接读读取结果。
            if args.use_local_rag and os.path.exists(args.save_path):
                # 读取 save_path 中的数据
                with open(args.save_path, "r") as file:
                    saved_data = json.load(file)

                task_to_retrieval = {
                    item["task_id"]: item["retrieval"] for item in saved_data
                }

                for sample in tqdm(testsets, desc="Processing samples"):
                    task_id = sample.get("task_id")

                    # 如果当前 sample 的 task_id 在保存的数据中，则使用保存的 retrieval 数据
                    if task_id in task_to_retrieval:
                        sample["rag"] = task_to_retrieval[task_id]

            else:
                retrieval = Retrieval(
                    retrieval=args.retrieval,
                    snippets_path=args.code_snippets_path,
                    prompt_version=args.prompt_version,
                    model_path=args.model_path,
                )

                retrieval_start_time = time.time()
                for sample in tqdm(testsets, desc="Retrieving code snippets"):
                    result = None
                    if args.retrieval == "summary":
                        query_code = sample["prefix"] + sample["suffix"]
                    elif args.fim_prompt == True:
                        query_code = retrieval.get_query_code_with_fim(
                            prefix=sample["prefix"], suffix=sample["suffix"]
                        )
                    elif args.oracle == True:
                        query_code = retrieval.get_query_code_with_oracle(
                            prefix=sample["prefix"],
                            suffix=sample["suffix"],
                            reference=sample["reference"],
                        )
                    else:
                        query_code = retrieval.get_query_code(
                            prefix=sample["prefix"], suffix=sample["suffix"]
                        )

                    if args.retrieval == "bandit":
                        # 这里可以直接给出results.
                        top_k, result = retrieval.get_bandit(sample["task_id"])
                    else:
                        top_k = retrieval.query_top_k(code=query_code, best_of=50)
                    sample["rag"] = top_k
                    sample["query_code"] = query_code
                    if result:
                        results.append(result)

                retrieval_end_time = time.time()
                print(
                    f"Retrieval avg time is {round((retrieval_end_time-retrieval_start_time)/len(testsets), 4)}"
                )
                if args.release_model:
                    retrieval.release_model()

                if args.retrieval == "bandit":
                    retrieval.runner.print_arm()

        context_data = []
        for testset in testsets:
            context = {
                "before_cursor": testset["prefix"],
                "after_cursor": testset["suffix"],
                "language": testset["language"],
                "path": testset["file_path"],
                "rag": testset.get("rag", None),
            }
            context_data.append(context)

        # 重新跑bandit
        # results = []
        if len(results) == 0:
            text_gen_server = TextGenerationServer(model_id=str(model_path))

            text_gen_client = TextGenerationClient(stop_sequences=stop_sequences)

            if args.model_name == "code_llama":
                interface = CodeLLamaInterface(
                    model_id=model_path,
                    total_budget=args.total_budget,
                    max_rag_num=args.max_rag_num,
                )
            elif args.model_name == "starcoder":
                interface = StarCoderInterface(model_id=model_path)
            else:
                raise NotImplementedError

            inputs = []
            for context in tqdm(context_data, desc="Generating prompts"):
                inputs.append(interface.gen_prompt(context))

            # Generate code results
            start_time = time.time()
            loop = asyncio.get_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(
                text_gen_client.generate_code_results(
                    inputs, args.max_new_tokens, num_outputs=1
                )
            )
            avg_time = round((time.time() - start_time) / len(inputs), 4)
        else:
            avg_time = 0

    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return -1

    df_data = []
    save_data = []
    type_metrics = {}  # 存储每种 type 的准确率数据

    for idx, (testset, output) in tqdm(
        enumerate(zip(testsets, results)),
        total=len(testsets),
        desc="Evaluating code generation results",
    ):
        output = output[0]
        reference: str = testset["reference"]

        if testset["type"] == "random_lines":
            reference_lines = reference.count("\n") + 1
            output = "\n".join(output.split("\n")[:reference_lines])

        edit_sim = ES(output, reference)
        exact_match = EM(output, reference)
        df_data.append((testset["language"], exact_match, edit_sim))
        save_data.append(
            {
                "task_id": testset["task_id"],
                "file_path": testset["file_path"],
                "line_ind": testset["line_ind"],
                "reference": reference,
                "output": output,
                "exact_match": exact_match,
                "edit_sim": edit_sim,
                "type": testset["type"],
                "retrieval": testset.get("rag", [])[:5],
                "query_code": testset.get("query_code", None),
            }
        )

        testset_type = testset["type"]
        if testset_type not in type_metrics:
            type_metrics[testset_type] = {"exact_match": 0, "edit_sim": 0, "count": 0}
        type_metrics[testset_type]["exact_match"] += exact_match
        type_metrics[testset_type]["edit_sim"] += edit_sim
        type_metrics[testset_type]["count"] += 1

    table = PrettyTable()
    table.field_names = ["Type", "Exact Match", "Edit Sim", "Count"]
    total_exact_match = total_edit_sim = total_count = 0

    for testset_type, data in type_metrics.items():
        count = data["count"]
        accuracy = round(data["exact_match"] / count, 4) if count else 0
        avg_edit_sim = round(data["edit_sim"] / count, 4) if count else 0
        table.add_row([testset_type, accuracy, avg_edit_sim, count])

        total_exact_match += data["exact_match"]
        total_edit_sim += data["edit_sim"]
        total_count += count

    overall_accuracy = round(total_exact_match / total_count, 4) if total_count else 0
    overall_avg_edit_sim = round(total_edit_sim / total_count, 4) if total_count else 0
    table.add_row(["Overall", overall_accuracy, overall_avg_edit_sim, total_count])

    print(table)

    # Calculate metrics
    df = pd.DataFrame(df_data, columns=["language", "exact_match", "edit_sim"])
    java_df = df.loc[df.language == "java"]
    java_em = round(java_df["exact_match"].mean(), 4)
    java_es = round(java_df["edit_sim"].mean(), 4)

    log_res = {
        "java_em": java_em,
        "java_es": java_es,
        "avg_time": avg_time,
        "testset_size": len(testsets),
    }
    logger.info(log_res)

    logger.info(f"Saving results to {args.save_path}")
    json.dump(save_data, open(args.save_path, "w"), indent=4)
    return 0


def main():
    args = parse_args()
    ret = run_eval_pipeline(args)
    sys.exit(ret)


if __name__ == "__main__":
    main()
