#!/bin/bash
#SBATCH --job-name=probe_hw
#SBATCH --partition=EPYC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=4gb
#SBATCH --time=00:05:00
#SBATCH -A dssc
#SBATCH --output=hardware_report.txt

# ------------------------------------------------------------------------
# Purpose: measure, on the ACTUAL node you will run on, every parameter
# that a tiling decision depends on. Don't trust datasheet numbers alone -
# ask the node directly, since cache/NUMA config can differ across the
# EPYC/THIN/GENOA partitions and even across BIOS settings.
# ------------------------------------------------------------------------

echo "======================================================================"
echo " Running on host: $(hostname)"
echo " Date: $(date)"
echo "======================================================================"

echo
echo "---------------- lscpu (sockets / cores / caches summary) ------------"
lscpu

echo
echo "---------------- Per-level cache sizes (ground truth) ----------------"
for cpu in /sys/devices/system/cpu/cpu0/cache/index*; do
    idx=$(basename "$cpu")
    level=$(cat "$cpu/level" 2>/dev/null)
    type=$(cat "$cpu/type" 2>/dev/null)
    size=$(cat "$cpu/size" 2>/dev/null)
    shared=$(cat "$cpu/shared_cpu_list" 2>/dev/null)
    line=$(cat "$cpu/coherency_line_size" 2>/dev/null)
    echo "  $idx : L${level} ${type}  size=${size}  line=${line}B  shared_by_cpus=${shared}"
done

echo
echo "---------------- NUMA topology -----------------------------------------"
numactl -H 2>/dev/null || echo "numactl not available"

echo
echo "---------------- lstopo (if available, gives CCX/CCD grouping) --------"
lstopo --of txt 2>/dev/null || echo "lstopo/hwloc not available - install with 'module load hwloc' if present"

echo
echo "======================================================================"
echo " Suggested tile sizes derived from the measured L1d size"
echo "======================================================================"
python3 - <<'EOF'
import os, glob, re

l1d_bytes = None
for idx in glob.glob("/sys/devices/system/cpu/cpu0/cache/index*"):
    lvl = open(idx + "/level").read().strip()
    typ = open(idx + "/type").read().strip()
    if lvl == "1" and typ == "Data":
        size_str = open(idx + "/size").read().strip()  # e.g. "32K"
        m = re.match(r"(\d+)K", size_str)
        if m:
            l1d_bytes = int(m.group(1)) * 1024

if l1d_bytes is None:
    print("Could not auto-detect L1d size, defaulting to 32KiB (typical Zen2 EPYC)")
    l1d_bytes = 32 * 1024

print(f"Measured/assumed L1d per core: {l1d_bytes/1024:.0f} KiB")

# A tile of side T touches, per stencil update, roughly one TxT block of
# 'old' plus one TxT block of 'new' (8 bytes/double each) that should stay
# resident in L1 across the inner loops for good reuse. Leave ~25% of L1
# free for the compiler's other live values (indices, halo rows, etc.)
usable = l1d_bytes * 0.75
import math
max_T = int(math.sqrt(usable / (2 * 8)))
print(f"Estimated max square tile side that fits in L1d: T <= {max_T}")

candidates = sorted(set(
    t for t in [8, 16, 32, 64, 128, 256] if True
))
print("Recommended sweep for the tiling benchmark (includes below/above the L1 estimate,")
print("plus larger tiles sized to fit L2 instead, to see the L1-vs-L2-vs-no-blocking picture):")
print(candidates)
EOF

echo
echo "Done. Use these numbers to pick TILE_SIZE candidates in compile_variants.sh"