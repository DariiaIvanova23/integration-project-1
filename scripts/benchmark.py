import time
import requests
import numpy as np

URL = "http://localhost:8000/predict"
FILE_PATH = "samples/yes.wav"

def run_benchmark():
    server_latencies = []
    client_latencies = []

    print("Виконуємо 10 розігрівних запитів...")
    for _ in range(10):
        with open(FILE_PATH, "rb") as f:
            requests.post(URL, files={"file": ("yes.wav", f, "audio/wav")})

    print("Виконуємо 100 тестових запитів...")
    for _ in range(100):
        t0 = time.perf_counter()
        with open(FILE_PATH, "rb") as f:
            resp = requests.post(URL, files={"file": ("yes.wav", f, "audio/wav")})
        t1 = time.perf_counter()

        client_latencies.append((t1 - t0) * 1000)
        server_latencies.append(resp.json()["latency_ms"])

    print("\n--- РЕЗУЛЬТАТИ БЕНЧМАРКУ (мс) ---")
    print(f"Сервер (Inference): Mean={np.mean(server_latencies):.2f} | p50={np.percentile(server_latencies, 50):.2f} | p95={np.percentile(server_latencies, 95):.2f}")
    print(f"Клієнт (End-to-End): Mean={np.mean(client_latencies):.2f} | p50={np.percentile(client_latencies, 50):.2f} | p95={np.percentile(client_latencies, 95):.2f}")

if __name__ == "__main__":
    run_benchmark()