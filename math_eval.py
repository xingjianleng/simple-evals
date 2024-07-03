"""
Measuring Mathematical Problem Solving With the MATH Dataset
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, Jacob Steinhardt
https://arxiv.org/abs/2103.03874
"""

import io
import os
import random
import re
import urllib

import pandas
import requests

from . import common
from .common import ANSWER_PATTERN, HTML_JINJA, check_equality
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

QUERY_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer to the problem.

{Question}

Remember to put your answer on its own line after "Answer:", and you do not need to use a \\boxed command.
""".strip()


def check_connectivity(url):
    try:
        response = requests.get(url)
        # Check if the server sent any response
        if response.ok or not response.ok:
            return True
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.RequestException as e:
        return False


class MathEval(Eval):
    def __init__(self, equality_checker: SamplerBase, file_path: str | None = None, num_examples: int | None = None):
        # Set the default file path to the math_test.csv file
        if file_path is None:
            file_path = "https://openaipublic.blob.core.windows.net/simple-evals/math_test.csv"
        if "http" in file_path:
            with urllib.request.urlopen(file_path) as f:
                df = pandas.read_csv(
                    io.BytesIO(f.read())
                )
        else:
            df = pandas.read_csv(
                file_path
            )
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples
        self.equality_checker = equality_checker

    def __call__(self, sampler: SamplerBase, num_threads: int, save_dir: str | None = None) -> EvalResult:
        # Create save_dir if it does not exist
        os.makedirs(save_dir, exist_ok=True)

        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(content=QUERY_TEMPLATE.format(**row), role="user")
            ]

            # Write the response to a file if save_dir is provided
            if save_dir:
                response_path = os.path.join(save_dir, f"response_{row['Unnamed: 0']}.txt")
                if os.path.exists(response_path):
                    with open(response_path, "r") as f:
                        response_text = f.read()
                else:
                    response_text = sampler(prompt_messages)
                    with open(response_path, "w") as f:
                        f.write(response_text)
            else:
                response_text = sampler(prompt_messages)

            if not check_connectivity(str(self.equality_checker.client.base_url)):
                return

            match = re.search(ANSWER_PATTERN, response_text)
            extracted_answer = match.group(1) if match else None
            score = float(check_equality(self.equality_checker, row["Answer"], extracted_answer))
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["Answer"],
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(html=html, score=score, convo=convo)

        results = common.map_with_progress(fn, self.examples, num_threads=num_threads)
        if any(result is None for result in results):
            return 
        return common.aggregate_results(results)
