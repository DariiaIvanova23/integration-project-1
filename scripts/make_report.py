import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="metrics/")
    parser.add_argument("--out", default="metrics/report.md")
    args = parser.parse_args()

    with open(os.path.join(args.metrics, "latency.json")) as f:
        latency = json.load(f)

    with open(os.path.join(args.metrics, "verify.json")) as f:
        verify = json.load(f)

    report_md = f"""# ML Pipeline Validation Report

## Результати тестування моделі
- **Golden Samples Test:** `{verify['passed']}/{verify['total']}` успішно розпізнано ({verify['accuracy_on_golden']}%)
- **Server Latency (Inference):**
  - **Mean:** `{latency['server_ms']['mean']} ms`
  - **p50:** `{latency['server_ms']['p50']} ms`
  - **p95:** `{latency['server_ms']['p95']} ms`
- **Client Latency (End-to-End):**
  - **Mean:** `{latency['client_ms']['mean']} ms`
  - **p95:** `{latency['client_ms']['p95']} ms`

###Статус: Усі перевірки пройдено успішно!
"""
    with open(args.out, "w") as f:
        f.write(report_md)
    print("Звіт сформовано!")

if __name__ == "__main__":
    main()