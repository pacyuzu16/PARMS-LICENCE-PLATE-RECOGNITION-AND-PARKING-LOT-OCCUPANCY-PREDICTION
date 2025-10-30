# utils.py
from pathlib import Path

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data" / "pklot_reduced"   # <-- CHANGED

def print_section(title: str):
    w = 70
    print("\n" + "="*w)
    print(f" {title} ".center(w))
    print("="*w + "\n")