"""
Evaluate the usefulness of context using an LLM critic.

"""
import concurrent.futures
import openai
import pickle
import re

from cik_benchmark import ALL_TASKS
from openai import OpenAI
from random import shuffle


client = OpenAI()

system_prompt = """
You are a critic whose role is to evaluate the quality of tasks
in the "context is key" time series forecasting benchmark.

"Context is Key" (CiK) is a time series forecasting benchmark 
that pairs numerical data with diverse types of carefully crafted
textual context, requiring models to integrate both modalities to
arrive at accurate predictions.

"""

task_prompt = """
Here is a task to evaluate.

<history>
{history}
</history>

<context>
    <background>
        {background}
    </background>
    <scenario>
        {scenario}
    </scenario>
    <constraints>
        {constraints}
    </constraints>
</context>

<future>
{future}
</future>

Assume the following two scenarios:
1) You are given only the numerical data in <history> and have no
additional information about the nature of the time series. You must
ignore the <context> section completely.

2) You are given the <context> section in addition to the numerical
data in <history>.

Now, assume you had to estimate the probability distribution of the
<future> values given the information available in each scenario. 
How would the quality of your estimation change in scenario 2 
compared to scenario 1?

First show your reasoning in <reason></reason> tags, then answer in 
<answer></answer> tags with either "much better", "slightly better", "unchanged", "worse" (no other reponses are allowed).
"""


def ts_to_str(data):
    return "\n".join([f"{k}: {v}" for k, v in data.items()])


def extract_html_tags(text, keys):
    """Extract the content within HTML tags for a list of keys.

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.

    Returns
    -------
    dict
        A dictionary mapping each key to a list of subset in `text` that match the key.

    Notes
    -----
    All text and keys will be converted to lowercase before matching.

    """
    content_dict = {}
    keys = set(keys)
    for key in keys:
        pattern = f"<{key}>(.*?)</{key}>"
        matches = re.findall(pattern, text, re.DOTALL)
        # print(matches)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


def process_task(task_cls, seed):
    print(task_cls.__name__)
    task = task_cls(seed=seed)
    print("...", task)

    # Assuming that when there is more than one series, we use the last one.
    history = ts_to_str(task.past_time.iloc[:, -1])
    future = ts_to_str(task.future_time.iloc[:, -1])

    # Ask the critic
    critique = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": task_prompt.format(
                    history=history, future=future, **task.__dict__
                ),
            },
        ],
        model="gpt-4o",
    )

    usage_info = critique.usage
    prompt_tokens = usage_info.prompt_tokens
    prompt_tokens_cached = usage_info.prompt_tokens_details.cached_tokens
    cost = (
        2.50 * (prompt_tokens - prompt_tokens_cached)
        + 1.25 * prompt_tokens_cached
        + 10 * usage_info.completion_tokens
    ) / 1e6

    output = extract_html_tags(
        critique.choices[0].message.content, ["reason", "answer"]
    )
    output["task"] = task_cls.__name__
    output["seed"] = seed
    output["reason"] = output["reason"][0]
    output["answer"] = output["answer"][0]
    output["cost"] = cost

    return output


if __name__ == "__main__":
    shuffle(ALL_TASKS)
    seeds = range(5)  # Adjust seeds if needed

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Prepare tasks for each task class and seed
        futures = [
            executor.submit(process_task, task_cls, seed)
            for task_cls in ALL_TASKS
            for seed in seeds
        ]

        # Collect results as they complete
        price = 0
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                critique = future.result()  # Process each result if needed
                price += critique["cost"]
                results.append(critique)
            except Exception as e:
                print(f"Task generated an exception: {e}")

    print(f"Total cost: ${price:.2f}")
    print(f"N. of results: {len(results)}")
    pickle.dump(results, open("llm_validation_results.pkl", "wb"))
