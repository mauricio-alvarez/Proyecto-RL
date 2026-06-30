import argparse
import json
import os
import urllib.request


DEFAULT_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", default="/workspace/data/raw/known_exploited_vulnerabilities.json")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    request = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()

    data = json.loads(payload.decode("utf-8"))
    with open(args.out, "wb") as handle:
        handle.write(payload)

    print("saved:", args.out)
    print("title:", data.get("title"))
    print("catalogVersion:", data.get("catalogVersion"))
    print("dateReleased:", data.get("dateReleased"))
    print("count:", data.get("count", len(data.get("vulnerabilities", []))))


if __name__ == "__main__":
    main()
