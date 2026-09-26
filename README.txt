###DRAGN-X pipeline 
##Mohamed Adil 
##12_24_25

#Prepare Your Input
- Edit config/samples.yaml to include paths to your percentage methylation files in .bw format. 
- Specify "male" or "female", if not specified the pipeline will skip SignalEnhancer step. 
- Set enhancement_factor to increase or decrease signal enhancement, recommended range 0-10. if 0 the pipeline will skip SignalEnhancer step. 
- Edit config/config.yaml to specify reference genome and methylation scale.

#Load Required Modules
ml snakemake/7.32.3-foss-2022b
ml Miniforge3/24.1.2-0

#Run Pipeline (remove -np after dry run)
snakemake -s DRAGN-X.snakefile --latency-wait 60 --keep-going --cluster-config config/cluster_slurm.yaml --cluster "sbatch -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus} -n {cluster.ntasks} -o {cluster.output} -J {cluster.JobName}" -j 40 --use-conda --verbose -p -np

#Output files
results/MethProfiler/* - extracted methylation files. These can be deleted for storage space.
results/Feature_matrix/* - file contains aggregated raw MethProfiler features used as input for DRAGN-X.
results/DRAGN-X/* file contains SignalEnhanced features, BaseModel predictions and DRAGN-X inferred expression for 18,172 genes. 

#Output annotation
SignalEnhanced features - Enhanced raw methylation features based on a reference background to increase signal to noise ratio
BaseModel - Series of models that predict probability of gene activity (0 - off, 1 - on). These models are marked as Model_1
DRAGN-X - Series of models that predict continuous gene expression (log2TPM+1 scale). These models are marked as Model_2 and Model_3


