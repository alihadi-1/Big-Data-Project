#!/usr/bin/env python3
import sys
import csv

reader = csv.reader(sys.stdin)
header = next(reader)  # skip header

for row in reader:
    try:
        close = float(row[4])  # Close price
        volume = int(row[5])
        print(f"close\t{close}")
        print(f"volume\t{volume}")
    except:
        continue
