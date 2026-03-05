import argparse

from transformers import AutoTokenizer

from learning_programs.attacks.common import CODE_GENERATION_SUPPORTED_MODELS, CODE_GENERATION_API_MODELS
from learning_programs.datasets.defect_detection import load_examples as load_examples_defect, Example as DefectExample
from learning_programs.datasets.clone_detection import load_examples as  load_examples_clone, Example as CloneExample
from learning_programs.attacks.code_generation.ours import Model, PASS_1


# This script does not support API models
SUPPORTED_MODELS = sorted(set(CODE_GENERATION_SUPPORTED_MODELS) - set(CODE_GENERATION_API_MODELS))


def get_prompt_defect(example: DefectExample) -> str:
    return f'''Analyze the following code snippet within the markdown code block and detect if the code is defective and insecure.

```
{example.code}
```

Answer 'Yes' if the code snippet has a defect and 'No' if it doesn't.
'''

def get_prompt_clone(example: CloneExample) -> str:
    return f'''Analyze the following two code snippets within the markdown code block and determine if the code snippets are clones or near clones of each other considering semantic equivalence.

#Code 1
```
{example.code1}
```

#Code 2
```
{example.code2}
```

Answer 'Yes' if the code snippets are clones or near-clones and 'No' if they are not.
'''


def finalize_prompt_defect(example: DefectExample, tokenizer: AutoTokenizer = None) -> str:
    _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"
    prompt = get_prompt_defect(example)
    chat_template = [
        {"role": "user", "content": prompt.lstrip()},
        {"role": "assistant", "content": f"The answer is:'{_MAGIC_SPLITTER_}'"}
    ]

    if tokenizer:
        return tokenizer.apply_chat_template(chat_template, tokenize=False).split(_MAGIC_SPLITTER_)[0]
    chat_template[1]["content"] = chat_template[1]["content"].split(_MAGIC_SPLITTER_)[0].rstrip() # Anthropic can't handle trailing whitespace
    return chat_template


def finalize_prompt_clone(example: CloneExample, tokenizer):
    _MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"
    prompt = get_prompt_clone(example)
    chat_template = [
        {"role": "user", "content": prompt.lstrip()},
        {"role": "assistant", "content": f"The answer is:'{_MAGIC_SPLITTER_}'"}
    ]

    if tokenizer:
        return tokenizer.apply_chat_template(chat_template, tokenize=False).split(_MAGIC_SPLITTER_)[0]
    chat_template[1]["content"] = chat_template[1]["content"].split(_MAGIC_SPLITTER_)[0].rstrip() # Anthropic can't handle trailing whitespace
    return chat_template


def parse_llm_resp(resp: str):
    return int(resp.startswith("Yes"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=SUPPORTED_MODELS, required=True)
    return parser.parse_args()


def main(args):
    print("Loading target model...")
    model = Model(args.model_name, None, True)

    print("Loading defect detection examples...")
    defect_set = load_examples_defect("test")
    defect_labels = [ex.label for ex in defect_set]
    defect_prompts = [finalize_prompt_defect(example, model.tokenizer) for example in defect_set]
    print("Generating defect detection predictions...")
    defect_req_outs = model.llm.generate(defect_prompts, sampling_params=PASS_1, use_tqdm=True)
    defect_pred_labels = [next(parse_llm_resp(o.text) for o in req_out.outputs) for req_out in defect_req_outs]
    defect_acc = sum(pred == label for pred, label in zip(defect_pred_labels, defect_labels)) / len(defect_labels)
    print(f"Accuracy: {defect_acc*100:.2f}%")

    print("Loading clone detection examples...")
    clone_set = load_examples_clone("test")
    clone_labels = [ex.label for ex in clone_set]
    clone_prompts = [finalize_prompt_clone(example, model.tokenizer) for example in clone_set]
    print("Generating clone detection predictions...")
    clone_req_outs = model.llm.generate(clone_prompts, sampling_params=PASS_1, use_tqdm=True)
    clone_pred_labels = [next(parse_llm_resp(o.text) for o in req_out.outputs) for req_out in clone_req_outs]
    clone_acc = sum(pred == label for pred, label in zip(clone_pred_labels, clone_labels)) / len(clone_labels)
    print(f"Accuracy: {clone_acc*100:.2f}%")


if __name__ == "__main__":
    main(parse_args())