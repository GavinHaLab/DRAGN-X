# DRAGN-X Pipeline

**Author:** Mohamed Adil &nbsp;·&nbsp; **Date:** December 24, 2025

A Snakemake pipeline that infers gene expression from DNA methylation profiles.

---

## 1. Download Required Files

Three directories are too large for GitHub and are hosted on [Zenodo](https://zenodo.org/records/22970009):

- **Reference_files**
- **Test_bw**
- **Trained_Models**

From inside the DRAGN-X pipeline directory, download and unzip them:

```bash
wget "https://zenodo.org/records/22970009/files/DRAGN-X_files.zip?download=1" -O DRAGN-X_files.zip
unzip DRAGN-X_files.zip
mv DRAGN-X_files/* .
rm -r DRAGN-X_files DRAGN-X_files.zip
```

This places all three directories inside the DRAGN-X pipeline directory.


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


## 6. DRAGN-X Annotation

Columns in the DRAGN-X output file:

| Column | Description | Sensitivity | Stability |
|---|---|---|---|
| name | Gene symbol | – | – |
| Model_1c2_Prob | BaseModel gene activity probability | – | – |
| Model_3c2_Prob_0_scaled | DRAGN-X inferred gene expression | High | Low |
| Model_3c2_Prob_100_scaled | DRAGN-X inferred gene expression | High | Low |
| Model_3c2_Prob_200_scaled | DRAGN-X inferred gene expression | Medium | Medium |
| Model_3c2_Prob_500_scaled | DRAGN-X inferred gene expression | Low | High |
| Model_3c2_Prob_1000_scaled | DRAGN-X inferred gene expression | Low | High |

Inferred gene expression is on the log2(TPM+1) scale.
