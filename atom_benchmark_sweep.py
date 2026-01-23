import argparse
import subprocess
import sys
import time
import requests
import os
import signal
import re
import csv
import itertools
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================

BENCHMARK_SCRIPT = "/app/bench_serving/benchmark_serving.py"

SWEEP_CONFIG = {
    "token_lengths": [                   # (Input, Output) tuples to sweep
        (1000, 1000),
        (4000, 1000),
        (10000, 1000),
    ],
    "max_concurrency": [
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
    ],
}

# Fixed Server Defaults (Can be overridden by args)
SERVER_DEFAULTS = {
    "host": "localhost",
    "port": 8000,
}

LLAMA_SERVER_CONFIG = {
    "max_num_seqs": 1024,
}

LLAMA_SERVER_ENV = {
    "VLLM_TORCH_PROFILER_WITH_STACK": "0",
}

QWEN3_SERVER_CONFIG = {
    "enable-expert-parallel": True,
    "max-model-len": "16384",
    "max-num-batched-tokens": "20000",
}

QWEN3_SERVER_ENV = {
    "AITER_QUICK_REDUCE_QUANTIZATION": "INT4",
    "ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION": "1",
}

# HELPER FUNCTIONS
# ==========================================

def get_server_command(args, server_config, server_env):
    """Constructs the server start command with environment variables."""
    
    # Environment variables
    env = os.environ.copy()
    for env_key, env_value in server_env.items():
        env[env_key] = str(env_value)

    cmd = [
        sys.executable, "-m", "atom.entrypoints.openai_server",
        "--model", args.model,
        "-tp", str(args.tp),
        "--kv_cache_dtype", "fp8",
        "--cudagraph-capture-sizes", "[1,2,4,8,16,30,32,62,64,128,160,192,256,320,512,640]",
        "--max-num-seqs", str(server_config["max_num_seqs"]),
    ]
    if "enable-expert-parallel" in server_config:
        cmd.append("--enable-expert-parallen")
    if "max-model-len" in server_config:
        cmd.append("--max-model-len")
        cmd.append(server_config["max-model-len"])
    if "max-num-batched-tokens" in server_config:
        cmd.append("--max-num-batched-tokens")
        cmd.append(server_config["max-num-batched-tokens"])
    
    return cmd, env

def wait_for_server(url, timeout=600):
    """Waits for the vLLM server to be ready."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        print(f"Waiting for server at {url}...")
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    print("Timeout waiting for server.")
    return False

def parse_metrics(output_str):
    """Parses stdout from benchmark_serving.py for key metrics."""
    metrics = {}
    
    # Regex patterns for standard vLLM benchmark output
    # Note: These patterns might need slight adjustment depending on exact vLLM version output
    output_token_throughput_pattern = re.compile(r"Output token throughput \(tok/s\):\s+([\d\.]+)")
    total_token_throughput_pattern = re.compile(r"Total Token throughput \(tok/s\):\s+([\d\.]+)")
    ttft_pattern = re.compile(r"Mean TTFT \(ms\):\s+([\d\.]+)")
    tpot_pattern = re.compile(r"Mean TPOT \(ms\):\s+([\d\.]+)")
    
    ott_match = output_token_throughput_pattern.search(output_str)
    ttt_match = total_token_throughput_pattern.search(output_str)
    ttft_match = ttft_pattern.search(output_str)
    tpot_match = tpot_pattern.search(output_str)
    
    if ott_match: metrics['output_token_throughput'] = float(ott_match.group(1))
    if ttt_match: metrics['total_token_throughput'] = float(ttt_match.group(1))
    if ttft_match: metrics['mean_ttft_ms'] = float(ttft_match.group(1))
    if tpot_match: metrics['mean_tpot_ms'] = float(tpot_match.group(1))
    
    return metrics

def run_benchmark_client(args, input_len, output_len, max_concurrency):
    """Runs the benchmark_serving.py client."""
    cmd = [
        sys.executable,
        "-m", "atom.benchmarks.benchmark_serving",
        "--backend", "vllm",
        "--host", str(args.host),
        "--port", str(args.port),
        "--model", args.model,
        "--dataset-name", "random",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--random-range-ratio", "1.0",
        "--max-concurrency", str(max_concurrency),
        "--num-prompts", str(max_concurrency * 10 if max_concurrency <= 128 else max_concurrency * 8),
        "--request-rate", "inf",
        "--percentile-metrics", "ttft,tpot",
        "--ignore-eos",
    ]

    print(f"cmd: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed: {e.stderr}")
        return None

# ==========================================
# COMMAND HANDLERS
# ==========================================

def handle_serve(args):
    """Handles the 'serve' subcommand."""
    if args.model == "amd/Llama-3.3-70B-Instruct-FP8-KV":
        server_config = LLAMA_SERVER_CONFIG
        server_env = LLAMA_SERVER_ENV
    elif args.model == "amd/Llama-3.3-70B-Instruct-MXFP4-Preview":
        server_config = LLAMA_SERVER_CONFIG
        server_env = LLAMA_SERVER_ENV
    elif args.model == "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8":
        server_config = QWEN3_SERVER_CONFIG
        server_env = QWEN3_SERVER_ENV
    else:
        sys.exit(1)

    cmd, env = get_server_command(args, server_config, server_env)
    print(f"Starting Server: {' '.join(cmd)}")
    
    process = subprocess.Popen(cmd, env=env)
    
    try:
        wait_for_server(f"http://{args.host}:{args.port}")
        process.wait() # Keep running until user kills it
    except KeyboardInterrupt:
        print("\nStopping server...")
        process.terminate()
        process.wait()

def handle_sweep(args):
    """Handles the 'sweep' subcommand."""
    results_file = f"sweep_results_tp{args.tp}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['TP', 'ISL', 'OSL', 'max_concurrency', 'num_prompts', 'output_token_throughput', 'total_token_throughput', 'mean_ttft_ms', 'mean_tpot_ms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate through Tensor Parallel sizes (requires restart)
        for input_len, output_len in SWEEP_CONFIG["token_lengths"]:
            for max_concurrency in SWEEP_CONFIG["max_concurrency"]:
                print(f"\n{'='*40}")
                print(f"Starting Config: input={input_len}, output={output_len}, max_concurrency={max_concurrency}")
                print(f"{'='*40}")
                # Iterate through Input/Output token lengths
                output = run_benchmark_client(args, input_len, output_len, max_concurrency)
                if output:
                    metrics = parse_metrics(output)
                    row = {
                        'TP': args.tp,
                        'ISL': input_len,
                        'OSL': output_len,
                        'max_concurrency': max_concurrency,
                        'num_prompts': max_concurrency * 10 if max_concurrency <= 128 else max_concurrency * 8,
                        'output_token_throughput': metrics.get('output_token_throughput', 0),
                        'total_token_throughput': metrics.get('total_token_throughput', 0),
                        'mean_ttft_ms': metrics.get('mean_ttft_ms', 0),
                        'mean_tpot_ms': metrics.get('mean_tpot_ms', 0)
                    }
                    writer.writerow(row)
                    print(f"Results: {row}")
                    csvfile.flush() # Ensure data is written
                    if metrics.get('mean_tpot_ms', 0) > 100:
                        break
                else:
                    print("No output collected for this run.")
    
    print(f"\nSweep complete. Results saved to {results_file}")

# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="atom Benchmark Sweep Script")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Shared Arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--model", required=True, help="HuggingFace model path")
    parent_parser.add_argument("--host", default=SERVER_DEFAULTS["host"], help="Server host")
    parent_parser.add_argument("--port", type=int, default=SERVER_DEFAULTS["port"], help="Server port")

    # Serve Command
    serve_parser = subparsers.add_parser("serve", parents=[parent_parser], help="Start vLLM Server")
    serve_parser.add_argument("--tp", type=int, default=1, help="TP Size for single serve")
    
    # Sweep Command
    sweep_parser = subparsers.add_parser("sweep", parents=[parent_parser], help="Run Benchmark Sweep")
    sweep_parser.add_argument("--tp", type=int, default=1, help="TP Size for single serve")

    args = parser.parse_args()

    if args.command == "serve":
        handle_serve(args)
    elif args.command == "sweep":
        handle_sweep(args)

if __name__ == "__main__":
    main()
