import os
import subprocess


# Makes the n GPUs with the most free memory (ties: least busy) the only visible ones.
# Call before importing torch. If CUDA_VISIBLE_DEVICES is already set (e.g., CUDA_VISIBLE_DEVICES=5 bash run_inputs.sh), it is kept.
def select_gpus(n):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"Using GPU(s) {os.environ['CUDA_VISIBLE_DEVICES']} (CUDA_VISIBLE_DEVICES set by the user)")
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # same GPU numbering as nvidia-smi
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    gpus = [[int(v) for v in line.split(",")] for line in out.strip().splitlines()]  # index, total, used, utilization
    gpus.sort(key=lambda g: (g[2] - g[1], g[3]))  # most free memory first, then least busy
    chosen = gpus[:n]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g[0]) for g in chosen)
    print("Using GPU(s) " + ", ".join(f"{g[0]} ({(g[1] - g[2]) / 1024:.1f} GiB free, {g[3]}% busy)" for g in chosen))
