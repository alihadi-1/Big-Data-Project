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

        if key == "Close":
            totals[key] += value
            counts[key] += 1
        elif key == "volume":
            maximum[key] = max(maximum[key], int(value))
    except:
        continue

if counts["Close"] > 0:
    avg_close = totals["Close"] / counts["Close"]

