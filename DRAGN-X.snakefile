#DRAGN-X.snakefile
#Mohamed Adil
#12-23-2025
#Ha & Haffner Lab
#Fred Hutchinson Cancer Research Center


#Before starting load
#ml snakemake/7.32.3-foss-2022b
#ml Miniforge3/24.1.2-0
"""
#command to run snakemake (remove -np at end when done validating):
snakemake -s DRAGN-X.snakefile --latency-wait 60 --keep-going --cluster-config config/cluster_slurm.yaml --cluster "sbatch -p {cluster.partition} --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus} -n {cluster.ntasks} -o {cluster.output} -J {cluster.JobName}" -j 40 --use-conda --verbose -p -np

#command to run snakemake on restart (remove -np at end when done validating):
#need to add '-q restart-new' according to scicomp. This should be fixed at some point.
snakemake -s DRAGN-X.snakefile --latency-wait 60 --keep-going --cluster-config config/cluster_slurm.yaml --cluster "sbatch -p restart-new -q restart-new --mem={cluster.mem} -t {cluster.time} -c {cluster.ncpus} -n {cluster.ntasks} -o {cluster.output} -J {cluster.JobName} --requeue" -j 40 -np
"""
import os  

configfile: "config/samples.yaml"
configfile: "config/sites.yaml"
configfile: "config/config.fred_hutch.yaml" 
configfile: "config/cluster_slurm.yaml" 

rule all:
	input:
		expand("{results_dir}/MethProfiler/{sites}/{samples}.Meth_profile.tsv",results_dir=config['results_dir'],samples=config['samples'], sites=config["sites"]),
		expand("{results_dir}/feature_matrix/{samples}.feature_matrix.tsv.gz",results_dir=config['results_dir'],samples=config['samples']),
		expand("{results_dir}/DRAGN-X/{samples}.DRAGN-X.results.tsv.gz",results_dir=config['results_dir'],samples=config['samples'])


rule generate_meth_profile:
	input:
		bw = lambda wildcards: config["samples"][wildcards.samples][0],
		bed = lambda wildcards: config["sites"][wildcards.sites]
	output:
		Meth_profile = "{results_dir}/MethProfiler/{sites}/{samples}.Meth_profile.tsv"
	params:
		sample_name = "{samples}",
		site_name = "{sites}",
		meth_profiler_script = config['code_dir']+'/Meth_profiler.py',
		results_dir=config['results_dir'] + "/MethProfiler",
		Reference_Genome = config['Reference_fasta'],
		Sliding_CpG_window = config['Sliding_CpG_window'],
		Sliding_distance = config['Sliding_distance'],
		CpG_thresholds_list = config['CpG_thresholds_list']
	conda:
		"envs/Meth_profiler_v1.yaml"
	shell:
		"time python {params.meth_profiler_script} \
		--in_file {input.bw} \
		--in_bed {input.bed} \
		--sample_name {params.sample_name} \
		--site_name {params.site_name} \
		--out_dir {params.results_dir} \
		--Reference_Genome {params.Reference_Genome} \
		--Sliding_CpG_window {params.Sliding_CpG_window} \
		--Sliding_distance {params.Sliding_distance} \
		--CpG_thresholds_list {params.CpG_thresholds_list}"

rule generate_feature_matrix:
	input: 
		meth_profiles = expand("{results_dir}/MethProfiler/{sites}/{samples}.Meth_profile.tsv",
		results_dir=config['results_dir'],
		sites=config['sites'],
		samples=["{samples}"])

	output:
		feature_matrix = "{results_dir}/feature_matrix/{samples}.feature_matrix.tsv.gz"
	params:
		sample_name = "{samples}",
		path_to_TSS_file = config['TSS_file'],
		Generate_feature_matrix_script = config['code_dir']+'/Generate_feature_matrix.py',
		#results_dir_for_script = lambda wildcards, input: os.path.dirname(input.meth_profiles[0])
		results_dir_for_script = lambda wildcards, input: os.path.join(config['results_dir'], "MethProfiler")
	conda:
		"envs/Meth_profiler_v1.yaml"
	shell:
		"time python -u {params.Generate_feature_matrix_script} \
		--sample_name {params.sample_name} \
    	--path_to_results {params.results_dir_for_script} \
		--path_to_TSS_file {params.path_to_TSS_file} \
		--path_to_save {output.feature_matrix}"


rule run_DRAGNX:
	input:
		feature_matrix = "{results_dir}/feature_matrix/{samples}.feature_matrix.tsv.gz"
	output:
		DRAGNX_file = "{results_dir}/DRAGN-X/{samples}.DRAGN-X.results.tsv.gz"
	params:
		DRAGN_X_script = config['code_dir']+'/DRAGNX.py',
		Trained_model = config['Trained_model_path'],
		Methylation_scale = config['Methylation_scale'],
		background_reference_Male = config['background_reference_Male_path'],
		background_reference_Female = config['background_reference_Female_path'],
		sample_sex = lambda wildcards: str(config["samples"][wildcards.samples][1]),
		enhancement_factor = lambda wildcards: str(config["samples"][wildcards.samples][2])
	conda:
		"envs/DRAGN-X_v1.yaml"
	shell:
		"time python -u {params.DRAGN_X_script} \
		--Path_to_trained_model {params.Trained_model} \
		--feature_matrix {input.feature_matrix} \
		--Methylation_scale {params.Methylation_scale} \
		--sample_sex {params.sample_sex} \
		--enhancement_factor {params.enhancement_factor} \
		--background_reference_Male {params.background_reference_Male} \
		--background_reference_Female {params.background_reference_Female} \
		--path_to_save {output.DRAGNX_file}"



