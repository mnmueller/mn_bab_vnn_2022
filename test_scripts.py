import argparse
import json
import os.path
import subprocess
import time
from typing import List, Tuple

from src.utilities.argument_parsing import get_config_from_json

TEMP_RUN_DIR = "run"


def load_instances(instance_path: str) -> List[Tuple[str, ...]]:
    with open(instance_path, "r") as f:
        lines = f.readlines()
    benchmarks: List[Tuple[str, ...]] = []
    for line in lines:
        benchmark = tuple([v.strip() for v in line.split(",")])
        benchmarks.append(benchmark)

    return benchmarks


def adapt_config(var: str, val: str) -> None:

    config_path = f"{TEMP_RUN_DIR}/config.json"
    config = get_config_from_json(config_path)
    config[var] = val

    with open(config_path, "w") as out:
        json.dump(config, out)


def run(instance_path: str, vary: List[str], benchmark: str) -> None:

    instances = load_instances(instance_path)
    out_path = "run/example_out.txt"
    # category, onnx, vnnlib, timeout
    for inst in instances:
        onnx, spec, timeout = inst[:3]
        onnx_path = f"./vnn-comp-2022-sup/benchmarks/{benchmark}/{onnx}"
        spec_path = f"./vnn-comp-2022-sup/benchmarks/{benchmark}/{spec}"
        start_time = time.time()
        if vary is not None and len(vary) > 1:
            attr = vary[0]
            vals = vary[1:]
            for v in vals:
                subprocess.call(
                    [
                        "bash",
                        "./prepare_instance.sh",
                        "v1",
                        benchmark,
                        onnx_path,
                        spec_path,
                    ]
                )
                adapt_config(attr, v)
                mid_time = time.time()
                subprocess.call(
                    [
                        "bash",
                        "./run_instance.sh",
                        "v1",
                        benchmark,
                        onnx_path,
                        spec_path,
                        out_path,
                        timeout,
                    ]
                )
        else:
            subprocess.call(
                ["bash", "./prepare_instance.sh", "v1", benchmark, onnx_path, spec_path]
            )
            mid_time = time.time()
            subprocess.call(
                [
                    "bash",
                    "./run_instance.sh",
                    "v1",
                    benchmark,
                    onnx_path,
                    spec_path,
                    out_path,
                    timeout,
                ]
            )
        end_time = time.time()
        if os.path.exists(out_path):
            with open(out_path, "r") as f:
                lines = f.readlines()
                if "unsat" in lines[0]:
                    result = "verified"
                elif "sat" in lines[0]:
                    result = "counterexample"
                else:
                    result = lines[0].strip()
        else:
            result = "NO OUTPUT FILE WRITTEN"

        print(
            f"Prepare time: {mid_time - start_time:.4f} | Runtime: {end_time - mid_time:.4f} | Timeout: {int(timeout):d} | Result: {result}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tests the given instances end to end. varying the provided parameter"
    )
    parser.add_argument(
        "-is", "--instances", type=str, help="The instance file path", required=True
    )

    parser.add_argument("-v", "--vary", type=str, nargs="*")

    args = parser.parse_args()

    benchmark = args.instances.split("/")[2]

    run(args.instances, args.vary, benchmark)
