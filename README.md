# DRAGN-X Pipeline

**Author:** Mohamed Adil &nbsp;·&nbsp; **Date:** December 24, 2025

A Snakemake pipeline that infers gene expression from DNA methylation profiles.

---

## 1. Download Required Files

Three directories are too large for GitHub and are hosted on [Zenodo](https://zenodo.org/records/22970009):

- **Reference_files**
- **Test_bw**
- **Trained_Models**

Download them and place all three inside the DRAGN-X pipeline directory. To download directly on the cluster:

```bash
pip install zenodo_get
zenodo_get 22970009
```

## 2. Prepare Your Input

**config/samples.yaml**
- Add paths to your percentage methylation files (.bw format).
- Specify sex as male or female. If not specified, the **SignalEnhancer** step is skipped.
- Set enhancement_factor to adjust signal enhancement (recommended range: 0–10). A value of 0 skips **SignalEnhancer**.

**config/config.yaml**
- Specify the reference genome and methylation scale.

## 3. Load Required Modules

```bash
ml snakemake/7.32.3-foss-2022b
ml Miniforge3/24.1.2-0
```

## 4. Run the Pipeline

Start with a dry run. Remove -np once it looks good.

```bash
snakemake -s DRAGN-X.snakefile \
  --latency-wait 60 \
  --keep-going \
  --cluster-config config/cluster_slurm.yaml \
  --cluster "sbatch -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus} -n {cluster.ntasks} -o {cluster.output} -J {cluster.JobName}" \
  -j 40 \
  --use-conda \
  --verbose \
  -p -np
```

## 5. Output Files

| Directory | Contents |
|---|---|
| results/MethProfiler/ | Extracted methylation files. Safe to delete to save space. |
| results/Feature_matrix/ | Aggregated raw MethProfiler features used as input for DRAGN-X. |
| results/DRAGN-X/ | SignalEnhanced features, BaseModel predictions, and DRAGN-X inferred expression for 18,172 genes. |

## 6. Output Annotation

| Term | Description |
|---|---|
| **SignalEnhanced features** | Raw methylation features enhanced against a reference background to increase signal-to-noise ratio. |
| **BaseModel** (Model_1) | Models predicting the probability of gene activity (0 = off, 1 = on). |
| **DRAGN-X** (Model_2, Model_3) | Models predicting continuous gene expression on a log2(TPM+1) scale. |
