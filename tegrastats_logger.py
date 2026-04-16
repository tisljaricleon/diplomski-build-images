"""
Logs tegrastats output to a CSV file.
Run on the Jetson alongside your tests:
    tegrastats --interval 1000 | python3 tegrastats_logger.py resource_log.csv

Or if tegrastats is already logging to a file:
    tegrastats --interval 1000 --logfile tegrastats.log
    python3 tegrastats_logger.py resource_log.csv < tegrastats.log
"""

import sys
import csv
import re

def parse_tegrastats_line(line):
    """Parse a single tegrastats output line into a dict."""
    data = {}

    # Timestamp: 04-16-2026 16:38:20
    ts_match = re.match(r'(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})', line)
    data['timestamp'] = ts_match.group(1) if ts_match else ''

    # RAM 3643/7620MB
    ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
    if ram_match:
        data['ram_used_mb'] = int(ram_match.group(1))
        data['ram_total_mb'] = int(ram_match.group(2))
    else:
        data['ram_used_mb'] = ''
        data['ram_total_mb'] = ''

    # SWAP 1/3810MB
    swap_match = re.search(r'SWAP (\d+)/(\d+)MB', line)
    if swap_match:
        data['swap_used_mb'] = int(swap_match.group(1))
        data['swap_total_mb'] = int(swap_match.group(2))
    else:
        data['swap_used_mb'] = ''
        data['swap_total_mb'] = ''

    # CPU [10%@729,4%@729,10%@729,11%@729,47%@729,66%@729]
    cpu_match = re.search(r'CPU \[([^\]]+)\]', line)
    if cpu_match:
        cpus = cpu_match.group(1).split(',')
        for i, cpu in enumerate(cpus):
            pct_match = re.match(r'(\d+)%@(\d+)', cpu.strip())
            if pct_match:
                data[f'cpu{i+1}_pct'] = int(pct_match.group(1))
                data[f'cpu{i+1}_freq'] = int(pct_match.group(2))
    
    # GR3D_FREQ 0%
    gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)
    data['gpu_util_pct'] = int(gpu_match.group(1)) if gpu_match else ''

    # Temperatures: cpu@45.125C soc2@45.937C etc
    for temp_name in ['cpu', 'gpu', 'soc0', 'soc1', 'soc2', 'tj']:
        temp_match = re.search(rf'{temp_name}@([\d.]+)C', line)
        data[f'temp_{temp_name}'] = float(temp_match.group(1)) if temp_match else ''

    # Power: VDD_IN 3680mW/3680mW VDD_CPU_GPU_CV 888mW/888mW VDD_SOC 1049mW/1049mW
    for pwr_name in ['VDD_IN', 'VDD_CPU_GPU_CV', 'VDD_SOC']:
        pwr_match = re.search(rf'{pwr_name} (\d+)mW/(\d+)mW', line)
        if pwr_match:
            data[f'{pwr_name.lower()}_cur_mw'] = int(pwr_match.group(1))
            data[f'{pwr_name.lower()}_avg_mw'] = int(pwr_match.group(2))

    return data


FIELDNAMES = [
    'timestamp',
    'ram_used_mb', 'ram_total_mb',
    'swap_used_mb', 'swap_total_mb',
    'cpu1_pct', 'cpu1_freq', 'cpu2_pct', 'cpu2_freq',
    'cpu3_pct', 'cpu3_freq', 'cpu4_pct', 'cpu4_freq',
    'cpu5_pct', 'cpu5_freq', 'cpu6_pct', 'cpu6_freq',
    'gpu_util_pct',
    'temp_cpu', 'temp_gpu', 'temp_soc0', 'temp_soc1', 'temp_soc2', 'temp_tj',
    'vdd_in_cur_mw', 'vdd_in_avg_mw',
    'vdd_cpu_gpu_cv_cur_mw', 'vdd_cpu_gpu_cv_avg_mw',
    'vdd_soc_cur_mw', 'vdd_soc_avg_mw',
]


def main():
    if len(sys.argv) < 2:
        print("Usage: tegrastats --interval 1000 | python3 tegrastats_logger.py output.csv")
        sys.exit(1)

    output_path = sys.argv[1]
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction='ignore')
        writer.writeheader()
        
        count = 0
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            data = parse_tegrastats_line(line)
            writer.writerow(data)
            f.flush()
            count += 1
            if count % 10 == 0:
                print(f"[TEGRASTATS] Logged {count} samples", end='\r')

    print(f"\n[TEGRASTATS] Done. Logged {count} samples to {output_path}")


if __name__ == "__main__":
    main()
