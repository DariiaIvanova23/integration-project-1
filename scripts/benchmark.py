import argparse
import json
import time
import requests
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", default="metrics/latency.json")
    args = parser.parse_args()

    file_path = "samples/yes.wav"

    for _ in range(args.warmup):
        with open(file_path, "rb") as f:
            requests.post(f"{args.url}/predict", files={"file": ("yes.wav", f, "audio/wav")})

    server_latencies, client_latencies = [], []

    for _ in range(args.n):
        t0 = time.perf_counter()
        with open(file_path, "rb") as f:
            resp = requests.post(f"{args.url}/predict", files={"file": ("yes.wav", f, "audio/wav")})
        t1 = time.perf_counter()

        client_latencies.append((t1 - t0) * 1000)
        server_latencies.append(resp.json()["latency_ms"])

    result = {
        "server_ms": {
            "mean": round(float(np.mean(server_latencies)), 2),
            "p50": round(float(np.percentile(server_latencies, 50)), 2),
            "p95": round(float(np.percentile(server_latencies, 95)), 2)
        },
        "client_ms": {
            "mean": round(float(np.mean(client_latencies)), 2),
            "p50": round(float(np.percentile(client_latencies, 50)), 2),
            "p95": round(float(np.percentile(client_latencies, 95)), 2)
        }
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("Бенчмарк завершено. Результати записано в:", args.out)

if __name__ == "__main__":
    main()