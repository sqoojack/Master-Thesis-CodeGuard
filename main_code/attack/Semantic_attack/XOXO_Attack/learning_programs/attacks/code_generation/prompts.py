CHAT_MODELS = {
    "Qwen/CodeQwen1.5-7B-Chat",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "mistralai/Codestral-22B-v0.1",
    "Artigenz/Artigenz-Coder-DS-6.7B",
    "m-a-p/OpenCodeInterpreter-DS-6.7B",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "openai/gpt-4.1",
    # "gpt-4o"
}

NON_CHAT_MODELS = {
    "meta-llama/Llama-3.2-3B-Instruct"
}

def prompt_template(task, tokenizer):
    _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"
    assert "## Example Start" in task
    problems = task.split("## Example Start")
    prompt = f'''
Please provide a self-contained Python script that solves the following problem in a markdown code block.

Consider the following functions found in the same project:

{problems[1].strip()}

{problems[2].strip()}

{problems[3].strip()}

Now write a function that solves the following problem.
Please use the same naming conventions and style as the functions above.
Please try to reuse the functions above if possible.
Pay attention to any additional global variables that may be defined in the project.

{problems[4].strip()}
'''

    response = f"""
Below is a self-contained Python script that solves the problem.
It uses the same naming conventions and style as the functions above.
It reuses the functions above where possible.
It also pays attention to any additional global variables that may be defined in the project.
```python
{_MAGIC_SPLITTER_}
```
"""
    chat_template = [
        {"role": "user", "content": prompt.lstrip()},
        {"role": "assistant", "content": response.lstrip()}
    ]

    if tokenizer:
        return tokenizer.apply_chat_template(chat_template, tokenize=False).split(_MAGIC_SPLITTER_)[0]
    chat_template[1]["content"] = chat_template[1]["content"].split(_MAGIC_SPLITTER_)[0].rstrip() # Anthropic can't handle trailing whitespace
    return chat_template



def get_few_shot_code(task: dict, problems: list[str]):
    return f'''
## Example Start
{problems[0].strip()}

## Example Start
{problems[1].strip()}

## Example Start
{problems[2].strip()}

## Example Start
{task}
'''