#!/usr/bin/env python3
import sys
from collections import defaultdict

totals = defaultdict(float)
counts = defaultdict(int)
maximum = defaultdict(int)

for line in sys.stdin:
    key, value = line.strip().split('\t')
    value = float(value)

    if key == "close":
        totals[key] += value
        counts[key] += 1
    elif key == "volume":
        maximum[key] = max(maximum[key], int(value))

# Output
if "close" in totals:
    avg = totals["close"] / counts["close"]
    print(f"Average Close:\t{avg:.2f}")
if "volume" in maximum:
    print(f"Max Volume:\t{maximum['volume']}")
