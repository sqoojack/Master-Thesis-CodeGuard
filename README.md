# CodeGuard: A Robust Defense Framework for Code LLMs

CodeGuard is a comprehensive framework designed to evaluate and defend Code Large Language Models (Code LLMs) against various malicious exploits, including Prompt Injection, Adversarial Attacks, and Semantic Attacks. 

## Features

* **Multi-Attack Simulation**: Simulates diverse attack vectors against Code LLMs, including Adversarial Attacks, Semantic Attacks, Adaptive Attacks, and XOXO Attacks.
* **Multi-layered Defense Mechanisms**: 
    * `Pre-filter`: Initial screening of inputs to detect obvious malicious payloads.
    * `Semantic Guardrail`: Analyzes the semantic intent of prompts to block indirect injections.
    * `Adversarial Guardrail`: Detects and neutralizes adversarial token perturbations.
    * `Dynamic Thresholding`: Adaptive security thresholds based on input context.
* **Comprehensive Evaluation Pipeline**: Automated scripts to run inferences, calculate Attack Success Rates (ASR), and evaluate model robustness using various datasets.

## Repository Structure

```text
master-thesis-codeguard/
├── Dataset/                     # Evaluation and training datasets
│   ├── Adaptive_attack/         # Contextual, copy-trigger, and decoy attacks
│   ├── ShadowCode/              # ShadowCode dataset and results
│   ├── Syntactic/               # Syntactic attack dataset
│   ├── XOXO_attack/             # XOXO clone and defect detection data
│   └── merged_all/              # Combined datasets for comprehensive testing
├── main_code/
│   ├── attack/                  # Implementations of different attack methodologies
│   │   ├── Adaptive_attack/
│   │   ├── Adversarial_attack/  # Includes INSEC, ShadowCode, etc.
│   │   └── Semantic_attack/     # Includes Flashboom, ITGen, XOXO_Attack
│   ├── defense/                 # Baseline defense implementations
│   ├── defense_v2/              # Advanced defense pipeline (CodeGuard Core)
│   │   ├── Adversarial_Guardrail.py
│   │   ├── Semantic_Guardrail.py
│   │   ├── dynamic_threshold.py
│   │   ├── pre_filter.py
│   │   └── main.py              # Main entry point for the defense pipeline
│   └── evaluate/                # Inference and evaluation scripts
│       ├── Inference_XOXO.py
│       ├── Inference_shadowcode.py
│       └── ...
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Installation

1. **Clone the repository:**
   ```
   git clone [https://github.com/your-username/master-thesis-codeguard.git](https://github.com/your-username/master-thesis-codeguard.git)
   cd master-thesis-codeguard
   ```

2. **Set up the environment:**
   It is recommended to use a virtual environment (e.g., `conda` or `venv`).
   ```
   python -m venv codeguard_env
   source codeguard_env/bin/activate  # On Windows use `codeguard_env\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

   *(Note: Some specific attack submodules like `Flashboom` or `ITGen` may have their own `requirements.txt` or `environment.yml` inside their respective directories. Please refer to them if you plan to run those specific attacks.)*

## Usage

### 1. Running the Defense Pipeline (CodeGuard)
To execute the multi-layered defense system against incoming prompts:

```
cd main_code/defense_v2
python main.py --input_data <path_to_input> --config <path_to_config>
```
*The defense pipeline sequentially routes the input through the `pre_filter`, `Semantic_Guardrail`, and `Adversarial_Guardrail`, applying the `dynamic_threshold` to determine if the prompt is safe for the LLM.*

### 2. Evaluating Attacks (Inference)
To evaluate how vulnerable a specific model is to a given attack dataset (e.g., ShadowCode):

```
cd main_code/evaluate
python Inference_shadowcode_v2.py --model_path <your_model_path> --dataset ../../Dataset/ShadowCode/shadowcode_dataset.jsonl
```

### 3. Generating Attacks
You can generate your own adversarial or semantic examples using the provided attack modules. For example, to generate semantic attacks via ITGen:

```
cd main_code/attack/Semantic_attack/ITGen/python_parser
bash build.sh # Build tree-sitter parsers first
cd ../CodeBERT_adv/Clone-detection/attack
python attack_itgen.py
```

## Datasets
The `Dataset/` directory contains pre-processed `.jsonl` and `.csv` files used for evaluation. Key datasets include:
* **ShadowCode**: Data poisoning and backdoor injection scenarios.
* **XOXO Attack**: Defect and clone detection bypass datasets.
* **Adaptive Attacks**: Datasets utilizing copy triggers and decoys to evade detection.