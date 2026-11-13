import argparse
import json
import sys
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--expected", default="samples/expected.json")
    parser.add_argument("--out", default="metrics/verify.json")
    args = parser.parse_args()

    with open(args.expected) as f:
        expected = json.load(f)

    passed, total = 0, len(expected)
    details = {}

    for filename, true_label in expected.items():
        filepath = f"samples/{filename}"
        with open(filepath, "rb") as f:
            resp = requests.post(f"{args.url}/predict", files={"file": (filename, f, "audio/wav")})
        
        if resp.status_code == 200:
            pred = resp.json().get("label")
            is_ok = (pred == true_label)
            if is_ok:
                passed += 1
            details[filename] = {"expected": true_label, "predicted": pred, "match": is_ok}
        else:
            details[filename] = {"error": resp.text, "match": False}

    summary = {"passed": passed, "total": total, "accuracy_on_golden": round(passed / total * 100, 2), "details": details}
    
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Golden samples: {passed}/{total} passed")
    if passed < total:
        print("Помилка: не всі еталонні зразки розпізнано вірно!")
        sys.exit(1)

if __name__ == "__main__":
    main()