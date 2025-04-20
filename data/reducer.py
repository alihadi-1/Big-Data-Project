#!/usr/bin/env python3
import sys
from collections import defaultdict

totals = defaultdict(float)
counts = defaultdict(int)
maximum = defaultdict(int)

for line in sys.stdin:
    try:
        key, value = line.strip().split('\t')
        value = float(value)

        if key == "close":
            totals[key] += value
            counts[key] += 1
        elif key == "volume":
            maximum[key] = max(maximum[key], int(value))
    except:
        continue

if counts["close"] > 0:
    avg_close = totals["close"] / counts["close"]
    print("Average Close\t%.2f" % avg_close)

if "volume" in maximum:
    print("Max Volume\t%d" % maximum["volume"])
