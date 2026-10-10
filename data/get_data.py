"""Download tiny shakespeare (~1MB) into data/input.txt"""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
out = Path(__file__).parent / "input.txt"

if out.exists():
    print(f"{out} already exists, skipping")
else:
    urllib.request.urlretrieve(URL, out)
    print(f"saved {out} ({out.stat().st_size / 1e6:.2f} MB)")
