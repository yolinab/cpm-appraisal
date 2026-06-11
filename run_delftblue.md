# Running on DelftBlue

All steps run on the **login node** unless noted. Compute nodes have no internet access.

## 1. Get the code onto DelftBlue

The repo is at `https://github.com/yolinab/cpm-appraisal`. The job script expects it at `~/cpm-appraisal`.

**First time:**
```bash
cd ~
git clone https://github.com/yolinab/cpm-appraisal.git
```

**Subsequent runs (pull latest changes):**
```bash
cd ~/cpm-appraisal
git pull
```

You do not need to be inside the project directory for most setup steps, but `conda env create -f environment-cluster.yml` must be run from inside the repo (or pass the full path to the file). `sbatch` must also be run from inside the repo so it can find `jobs/run_experiment.sh`.

---

## 2. Check if the conda environment already exists

```bash
conda env list
```

Look for `cpm` in the output. If it is there, skip creation. If it is missing or you want to update it after changing `environment-cluster.yml`:

```bash
# Create (first time):
conda env create -f ~/cpm-appraisal/environment-cluster.yml

# Update (environment exists but dependencies changed):
conda env update -f ~/cpm-appraisal/environment-cluster.yml --prune
```

---

## 3. Check if Qwen is already downloaded

The model is cached under `HF_HOME`. The job script sets this to `/scratch/$USER/hf_cache`.

```bash
ls /scratch/$USER/hf_cache/hub/ | grep -i qwen
```

If you see a directory like `models--Qwen--Qwen2.5-7B-Instruct`, it is already downloaded. If the directory is empty or missing, download it now (on the login node, which has internet):

```bash
export HF_HOME=/scratch/$USER/hf_cache
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
```

This takes a few minutes (~15 GB). You only need to do it once; the cache persists on `/scratch`.

> **Note:** `/scratch` is not backed up and may be purged after inactivity. If the cache is gone, re-run the download step.

---

## 4. Check if the logs directory exists

```bash
ls ~/cpm-appraisal/logs
```

If you get `No such file or directory`, create it:

```bash
mkdir -p ~/cpm-appraisal/logs
```

The job script creates `data/outputs/` itself via `mkdir -p`, but not `logs/`. SLURM will silently fail to write stdout/stderr if the directory is missing.

---

## 5. Submit the job

From inside the project directory:

```bash
cd ~/cpm-appraisal
sbatch jobs/run_experiment.sh
```

SLURM will print a job ID, e.g. `Submitted batch job 12345`.

**Monitor the job:**
```bash
squeue -u $USER
```

**Watch live output:**
```bash
tail -f ~/cpm-appraisal/logs/12345.out
```

**Check for errors:**
```bash
cat ~/cpm-appraisal/logs/12345.err
```

Results are written to `data/outputs/results_qwen.json` when the job completes.

