#!/bin/bash

# Create default mapper.py
cat <<EOF > /data/mapper.py
#!/usr/bin/env python3
import sys
import csv
reader = csv.reader(sys.stdin)
header = next(reader, None)
for row in reader:
    try:
        print(f"close\\t{float(row[4])}")
        print(f"volume\\t{int(row[5])}")
    except:
        continue
EOF

# Create default reducer.py
cat <<EOF > /data/reducer.py
#!/usr/bin/env python3
import sys
from collections import defaultdict
totals = defaultdict(float)
counts = defaultdict(int)
maximum = defaultdict(int)
for line in sys.stdin:
    key, value = line.strip().split('\\t')
    value = float(value)
    if key == "close":
        totals[key] += value
        counts[key] += 1
    elif key == "volume":
        maximum[key] = max(maximum[key], int(value))
if "close" in totals:
    avg = totals["close"] / counts["close"]
    print(f"Average Close:\\t{avg:.2f}")
if "volume" in maximum:
    print(f"Max Volume:\\t{maximum['volume']}")
EOF

chmod +x /data/mapper.py /data/reducer.py
