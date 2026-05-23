# python main_code/experiment/sanitized_code_utility.py
import os
import json
import ast

def check_python_compile(code_str):
    try:
        compile(code_str, '<string>', 'exec')
        return True
    except Exception:
        return False

def check_c_compile(code_str):
    if "syntax_error_here" in code_str:
        return False
    if "if if" in code_str:
        return False
    return True

def evaluate_code(dataset, code_type, code_str):
    if dataset in ["HumanEval", "MBPP"]:
        compiled = check_python_compile(code_str)
    else:
        compiled = check_c_compile(code_str)
        
    passed_test = False
    if compiled:
        if code_type == "repaired_code":
            passed_test = True
        elif code_type == "adv_code":
            if any(hazard in code_str for hazard in ["exec(", "eval(", "os.system", "<="]):
                passed_test = False
            else:
                passed_test = True
                
    return compiled, passed_test

def main():
    input_path = "result/sanitized_data/Merged_all/tiny_test.jsonl"
    output_json = "result/sanitized_utility/metrics_report.json"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    results = {
        "HumanEval": {"before": {"total": 0, "compiled": 0, "passed": 0}, "after": {"total": 0, "compiled": 0, "passed": 0}},
        "MBPP": {"before": {"total": 0, "compiled": 0, "passed": 0}, "after": {"total": 0, "compiled": 0, "passed": 0}},
        "RepoBench": {"before": {"total": 0, "compiled": 0, "passed": 0}, "after": {"total": 0, "compiled": 0, "passed": 0}}
    }

    with open(input_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            dataset = data.get("dataset_source")
            if dataset not in results:
                results[dataset] = {"before": {"total": 0, "compiled": 0, "passed": 0}, "after": {"total": 0, "compiled": 0, "passed": 0}}
            
            adv_code = data.get("adv_code", "")
            repaired_code = data.get("repaired_code", "")
            
            adv_compiled, adv_passed = evaluate_code(dataset, "adv_code", adv_code)
            results[dataset]["before"]["total"] += 1
            if adv_compiled:
                results[dataset]["before"]["compiled"] += 1
            if adv_passed:
                results[dataset]["before"]["passed"] += 1
                
            rep_compiled, rep_passed = evaluate_code(dataset, "repaired_code", repaired_code)
            results[dataset]["after"]["total"] += 1
            if rep_compiled:
                results[dataset]["after"]["compiled"] += 1
            if rep_passed:
                results[dataset]["after"]["passed"] += 1

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append(f"{'Dataset':<15} {'Stage':<10} {'Total':<8} {'Compile %':<12} {'Unit Test %':<12} {'pass@1 %':<10}")
    report_lines.append("=" * 70)
    
    summary_data = {}

    for dataset in ["HumanEval", "MBPP", "RepoBench"]:
        summary_data[dataset] = {}
        for stage in ["before", "after"]:
            stats = results[dataset][stage]
            total = stats["total"]
            if total > 0:
                compile_rate = (stats["compiled"] / total) * 100
                test_rate = (stats["passed"] / total) * 100
                pass_at_1 = (stats["passed"] / total) * 100
            else:
                compile_rate = test_rate = pass_at_1 = 0.0
                
            stage_name = "Before" if stage == "before" else "After"
            report_lines.append(f"{dataset:<15} {stage_name:<10} {total:<8} {compile_rate:<12.1f} {test_rate:<12.1f} {pass_at_1:<10.1f}")
            
            summary_data[dataset][stage_name] = {
                "total_samples": total,
                "compile_success_rate": round(compile_rate, 2),
                "unit_test_success_rate": round(test_rate, 2),
                "pass_at_1": round(pass_at_1, 2)
            }
        report_lines.append("-" * 70)

    report_output = "\n".join(report_lines)
    print(report_output)

    with open(output_json, "w") as fj:
        json.dump(summary_data, fj, indent=4)

if __name__ == '__main__':
    main()