import sys
import argparse
import os

# Resolve path issues so it can find defense module and its dependencies
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
defense_dir = os.path.join(project_root, "defense")
if defense_dir not in sys.path:
    sys.path.append(defense_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from main import (
    build_arg_parser,
    apply_params_file,
    load_model_and_tokenizer,
    build_guardrails,
    run_pipeline
)

def main():
    """CLI wrapper allowing extensions to execute python framework logic via subprocess pipes."""
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--lang", type=str, default="c")
    parsed_args, remainder = cli_parser.parse_known_args()

    # Build primary core defense configuration objects
    base_parser = build_arg_parser()
    args = base_parser.parse_args(["--attack_type", "default_attack", "--mode", "all"])
    try:
        args = apply_params_file(args)
    except Exception:
        pass

    # Instantiate runtime objects
    model, tokenizer, device = load_model_and_tokenizer(args)
    guardrails = build_guardrails(args, model, tokenizer, device)

    # Ingest text streams from sys.stdin pipelines
    input_code = sys.stdin.read()
    pipeline_result = run_pipeline(input_code, parsed_args.lang, guardrails, args)
    
    # Emit sanitized data onto standard stdout channel
    sys.stdout.write(pipeline_result.get("final_code", input_code))
    sys.stdout.flush()

if __name__ == "__main__":
    main()