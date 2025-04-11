#!/usr/bin/env python3
import sys
import csv

reader = csv.reader(sys.stdin)
header = next(reader, None)

for row in reader:
    try:
        close = float(row["Close"])
        volume = int(row["Volume"])
    except:
        continue
