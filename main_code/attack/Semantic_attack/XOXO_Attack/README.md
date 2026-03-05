# GCGS Attack Reproducibility Package
We include instructions to fully reproduce our paper's results below. Please refer to the paper's appendix for detailed information regarding the hardware, driver, and OS environment we used.

## Installing the required Linux packages
Run the following command to install the required packages:

```bash
sudo apt install build-essential python-is-python3 python3-dev
```

## Generating random seeds
To maximize reproducibility, we seed all random number generators with `2024`. To generate the random seeds to finetune the target models and run the attacks, we do the following:
```bash
python -m learning_programs.reproduce.get_seeds --seed 2024 --num_seeds 5
```
The generated seeds are:
```
2019024262
1304602151
2140188135
4137125744
3023913752
```

## Reproducing code reasoning results
Please follow the instructions below to set up your virtual environment:
```bash
python -m venv ./.venv_reasoning
source ./.venv_reasoning/bin/activate
pip install -r requirements/reasoning.txt
```

Additionally, specify the GPUs to use by setting the `CUDA_VISIBLE_DEVICES` environment variable correspondingly.

### Running all attacks
To reproduce all results, please run the following:
```bash
python -m learning_programs.reproduce.run --all --datasets defect_detection clone_detection
```
The aggregated results for each `dataset` can then be found in `results/attacks/paper/[defect_detection | clone_detection]_results.csv`. These results correspond to Table 2 in the paper.

### Running a single attack
To reproduce a single result for code reasoning, follow these instructions:
```bash
export MODEL=[microsoft/codebert-base | microsoft/graphcodebert-base | Salesforce/codet5p-110m-embedding]
export DATASET=[defect_detection | clone_detection]
export SEED=[2019024262 | 1304602151 | 2140188135 | 4137125744 | 3023913752]
export ATTACK=[mhm | alert | rnns | wir | ours | ours_trained] # ours = GCGS; ours_trained GCGS+W
```
1. Preprocess the dataset:
```bash
python -m learning_programs.datasets.preprocess --$DATASET
```
2. Finetune a target model:
```bash
python -m learning_programs.attacks.target --finetune --dataset $DATASET --model_name $MODEL --seed $SEED
```
3. Test the target model:
```bash
python -m learning_programs.attacks.target --test --dataset $DATASET --model_name $MODEL --seed $SEED
```
4. Run an attack:
```bash
python -m learning_programs.attacks.$DATASET.$ATTACK --model_name $MODEL --seed $SEED
```
5. Print results
```bash
python -m learning_programs.attacks.print_results --compute_codebleu --dataset $DATASET --attack $ATTACK --model_name $MODEL --seed $SEED
```

## Reproducing in-context buggy/vulnerable code generation results
Please follow the instructions below to set up your virtual environment:
```bash
python -m venv ./.venv_generation
source ./.venv_generation/bin/activate
pip install -r requirements/generation.txt
```
Additionally:
- To run `GPT 4.1` using OpenRouter, set the `OPENROUTER_API_KEY` environment variable. See OpenRouter documentation for more detail.
- To run `Claude 3.5 Sonnet v2 using GCP Vertex AI API`, set `GCP_CRED_FILE`, `PROJECT_ID`, and `MODEL_LOCATION` environment variables. See GCP Vertex AI API documentation for more detail.
- To run `CWEval` experiments, you need to have `docker` installed and the user needs to have the correct permissions to use `docker`. See Docker documentation for more details.
- To run some models, you need to be logged in to the Hugging Face Hub and have the permission to use the corresponding models. Please see the Hugging Face Hub documentation for more details.
- Specify the GPUs to use by setting the `CUDA_VISIBLE_DEVICES` environment variable correspondingly.

### Running all attacks
To reproduce all results, please run the following:
```bash
python -m learning_programs.reproduce.run --all --datasets humaneval mbpp cweval
```
The aggregated results for each `dataset` can then be found in `results/attacks/paper/[humaneval | mbpp | cweval]_results.csv`. These results correspond to Table 1 in the paper.

### Running a single attack
To reproduce a single result for code generation, follow these instructions:
```bash
export MODEL=[Qwen/Qwen2.5-Coder-7B-Instruct | Qwen/Qwen2.5-Coder-32B-Instruct | mistralai/Codestral-22B-v0.1 | meta-llama/Llama-3.1-8B-Instruct | deepseek-ai/deepseek-coder-6.7b-instruct | deepseek-ai/deepseek-coder-33b-instruct | openai/gpt-4.1 | claude-3-5-sonnet-v2@20241022]
export DATASET=[humaneval | mbpp | cweval]
export SEED=[2019024262 | 1304602151 | 2140188135 | 4137125744 | 3023913752]
export ATTACK=[ours | ours_no_logprobs | ours_limited] # ours_no_logprobs = GCGS; ours = GCGS+P; ours_limited = a single limited run on API models
```
1. Preprocess the `CodeSearchNet/Python` dataset that we use to extract replacement identifiers from:
```bash
python -m learning_programs.datasets.preprocess --summarization
```
2. Run an attack:
```bash
python -m learning_programs.attacks.$DATASET.$ATTACK --model_name $MODEL --seed $SEED
```
5. Print results
```bash
python -m learning_programs.attacks.print_results --compute_codebleu --dataset $DATASET --attack $ATTACK --model_name $MODEL --seed $SEED
```
