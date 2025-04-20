#!/usr/bin/env python3
import sys
import csv

reader = csv.reader(sys.stdin)
header = next(reader, None)

for row in reader:
    try:
        close = float(row[1])   # Close price
        volume = int(row[5])    # Volume
        print("close\t%.2f" % close)
        print("volume\t%d" % volume)
    except:
        continue
