import argparse
import json
import os
import time

import pandas as pd

from . import common
from .math_eval import MathEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)


LLM_MODELS = {
    # vllm models:
    "llama3-8b_vllm_assistant": ChatCompletionSampler(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        base_url="http://localhost:8000/v1",
    ),
    "llama3-8b_vllm_chatgpt": ChatCompletionSampler(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        base_url="http://localhost:8000/v1",
    ),
    "llama3-70b_vllm_assistant": ChatCompletionSampler(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        base_url="http://localhost:8000/v1",
    ),
    "llama3-70b_vllm_chatgpt": ChatCompletionSampler(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        base_url="http://localhost:8000/v1",
    ),
    "qwen2-7b_vllm_assistant": ChatCompletionSampler(
        model="Qwen/Qwen2-7B-Instruct",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        base_url="http://localhost:8000/v1",
    ),
    "qwen2-7b_vllm_chatgpt": ChatCompletionSampler(
        model="Qwen/Qwen2-7B-Instruct",
        system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        base_url="http://localhost:8000/v1",
    ),
    "qwen2-72b_vllm_assistant": ChatCompletionSampler(
        model="Qwen/Qwen2-72B-Instruct",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        base_url="http://localhost:8000/v1",
    ),
    "qwen2-72b_vllm_chatgpt": ChatCompletionSampler(
        model="Qwen/Qwen2-72B-Instruct",
        system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        base_url="http://localhost:8000/v1",
    ),
    # chatgpt models:
    "gpt-4-turbo-2024-04-09_assistant": ChatCompletionSampler(
        model="gpt-4-turbo-2024-04-09",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
    ),
    "gpt-4-turbo-2024-04-09_chatgpt": ChatCompletionSampler(
        model="gpt-4-turbo-2024-04-09",
        system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ),
    "gpt-4o_assistant": ChatCompletionSampler(
        model="gpt-4o",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
    ),
    "gpt-4o_chatgpt": ChatCompletionSampler(
        model="gpt-4o",
        system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        max_tokens=2048,
    ),
}


def main(args):
    sampler = LLM_MODELS[args.llm]
    equality_checker = ChatCompletionSampler(model=args.checker_llm)

    evaluator = MathEval(equality_checker=equality_checker,
                         file_path=args.file_path,
                         num_examples=args.num_examples if args.num_examples > 0 else None)

    # The output directory
    output_path = os.path.join(
        os.path.dirname(__file__),
        "output",
        f"{os.path.basename(args.file_path).split('.')[0]}_{args.llm}")
    os.makedirs(output_path, exist_ok=True)

    result = evaluator(sampler, num_threads=args.num_threads, save_dir=output_path)
    if result is None:
        print("=== Cannot reach the Evaluator LLM, double check! ===")
        return

    # ^^^ how to use a sampler
    report_filename = os.path.join(output_path, f"report_{args.checker_llm}.html")
    print(f"Writing report to {report_filename}")
    with open(report_filename, "w") as fh:
        fh.write(common.make_report(result))
    metrics = result.metrics | {"score": result.score}
    print(metrics)
    result_filename = os.path.join(output_path, f"report_{args.checker_llm}.json")

    # Perform evaluation only if the file does not exist
    with open(result_filename, "w") as f:
        f.write(json.dumps(metrics, indent=2))
    print(f"Writing results to {result_filename}")

    print("=== Done! ===")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser("MATH dataset evaluation")
    parser.add_argument("--file_path", type=str, required=True,
                        help="The path to the math_test.csv file")
    parser.add_argument("--llm", type=str, required=True,
                        help="The LLM model to use for evaluation, separated by a comma")
    parser.add_argument("--checker_llm", type=str, default="gpt-4-turbo-preview",
                        help="The LLM model to use for checking equality")
    parser.add_argument("--num_examples", type=int, default=-1,
                        help="Number of examples to evaluate")
    parser.add_argument("--num_threads", type=int, default=2,
                        help="Number of threads to use for evaluation")
    args = parser.parse_args()

    # Start evaluation
    main(args)
