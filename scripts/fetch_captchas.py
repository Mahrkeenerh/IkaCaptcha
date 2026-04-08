"""
Fetch raw captcha images from the game server. No labeling.

Usage:
    python scripts/fetch_captchas.py --count 9000 --cookie "100597_abc123..."
"""

import argparse
import random
import time
from pathlib import Path

import requests
from io import BytesIO
from PIL import Image

CAPTCHA_URL = "https://s53-cz.ikariam.gameforge.com/?action=Options&function=createCaptcha&rand={rand}"
DEFAULT_OUTPUT = "data/real_samples_unlabeled"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=9000)
    parser.add_argument("--cookie", type=str, required=True,
                        help="Value of the 'ikariam' session cookie")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    existing = len(list(out.glob("*.png")))
    print(f"Output: {out}/ ({existing} existing)")

    cookies = {"ikariam": args.cookie}
    fetched = 0
    failed = 0

    for i in range(args.count):
        idx = existing + i
        try:
            rand = random.randint(1, 2_000_000_000)
            url = CAPTCHA_URL.format(rand=rand)
            resp = requests.get(url, cookies=cookies, timeout=10)
            resp.raise_for_status()

            if "image" not in resp.headers.get("content-type", ""):
                print(f"  [{idx}] Not an image — session expired?")
                failed += 1
                if failed > 10:
                    print("Too many failures, stopping. Check your cookie.")
                    break
                continue

            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(out / f"{idx:05d}.png")
            fetched += 1

            if fetched % 100 == 0:
                print(f"  {fetched}/{args.count} fetched ({failed} failed)")

        except Exception as e:
            print(f"  [{idx}] Error: {e}")
            failed += 1
            if failed > 10:
                print("Too many failures, stopping.")
                break

        time.sleep(random.uniform(1.0, 2.5))

    print(f"\nDone: {fetched} fetched, {failed} failed")
    print(f"Total in {out}/: {existing + fetched}")


if __name__ == "__main__":
    main()
