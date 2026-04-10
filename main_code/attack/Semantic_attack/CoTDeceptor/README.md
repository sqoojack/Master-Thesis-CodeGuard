# Code Obfuscation Framework

A strategy-tree-based code obfuscation tool that iteratively applies transformations to source code, evaluates the results, and reflects on strategies based on a research paper's methodology.

## Getting Started

### Prerequisites
* **Conda**: Ensure you have Miniconda or Anaconda installed.
* **API Key**: A valid Groq API key.

### Installation
1.  **Create and activate the environment:**
    ```bash
    conda create -n obfuscator python=3.10
    conda activate obfuscator
    ```
2.  **Install dependencies:**
    ```bash
    pip install groq numpy
    ```

### Configuration
1.  Open `main.py`.
2.  Update the `api_key` variable with your credentials.
3.  Adjust the `time_sleep` function settings if necessary to manage rate limits.

### Execution
Run the main process:
```bash
python main.py
```

---

## File Structure & Responsibilities

| File | Description |
| :--- | :--- |
| `main.py` | Entry point. Initializes classes and executes the workflow according to the paper's logic. |
| `obfuscator.py` | Core engine that applies selected strategies from `strategy.py` to the raw code. |
| `strategy.py` | Repository of all available obfuscation strategies and their implementations. |
| `tree.py` | Manages the construction and navigation of the strategy tree. |
| `verifier.py` | Evaluates and scores the processed code based on predefined metrics. |
| `reflector.py` | Analyzes feedback (scores/descriptions) from the verifier to guide future iterations. |
| `utils.py` | Contains shared utility functions and helper methods. |
| `playbook.json` | Logs the results and transformations of each iteration for tracking. |

---

## How to Add New Strategies

To extend the framework with custom obfuscation logic, follow these steps in `strategy.py`:

1.  **Register the strategy**: 
    Add a new entry to the `_register_builtin(self)` method using the `Strategy` object.
    ```python
    self.strategies["new_strategy_name"] = Strategy(
        "new_strategy_name", 
        "category", 
        "Description of strategy", 
        self._new_strategy_func
    )
    ```

2.  **Implement the function**: 
    Define the corresponding private method `_new_strategy_func` within the class to handle the code transformation logic.