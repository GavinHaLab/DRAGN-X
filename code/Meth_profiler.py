#MA_23_101
#ml Python/3.9.6-GCCcore-11.2.0

import os
import time
import pandas as pd
import numpy as np
import pyBigWig
import argparse
from multiprocessing import Pool
import psutil
from Bio import SeqIO
from Bio.Seq import Seq
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--in_file', help='path to bw file', required=True)
parser.add_argument('--in_bed', help='path to bed file', required=True)
parser.add_argument('--sample_name', help='name for sample on output file', required=True)
parser.add_argument('--site_name', help='name for site on output file', required=True)
parser.add_argument('--out_dir', help='dir where output files should be saved', required=True)
parser.add_argument('--Reference_Genome', help='Reference genome fasta file', required=True)
parser.add_argument('--Sliding_CpG_window', help='Sliding window size for calculating methylation based on CpG_threshold', required=True)
parser.add_argument('--Sliding_distance', help='Sliding steps for sliding_CpG_window', required=True)
parser.add_argument('--CpG_thresholds_list', help='CpG thresholds for sliding window', required=True,nargs='+')



args = parser.parse_args()

in_file = args.in_file
in_bed = args.in_bed

out_name = args.sample_name

site_name = args.site_name

current_bed = pd.read_csv(in_bed,sep = '\t', header = None)
current_bed.columns = ["Chrom","Start","End","name","score","strand"] + current_bed.columns[6:].to_list()

out_dir = args.out_dir

print("Num of sites with coord", len(current_bed))

flank_length = 0
genome_fasta = args.Reference_Genome
genome = SeqIO.to_dict(SeqIO.parse(genome_fasta, "fasta"))

#cpu
#num_cpus = psutil.cpu_count(logical=False)
num_cpus = 1

#load bw 
bw = pyBigWig.open(in_file)

#Sliding window params
Window_size = int(args.Sliding_CpG_window)
slide_distance = int(args.Sliding_distance)
CpG_thresholds_list = [float(val) for val in args.CpG_thresholds_list]

Sliding_vals_name = []
for each_threshold in CpG_thresholds_list:
    CpG_density = each_threshold
    Sliding_vals_name = Sliding_vals_name + ["mean_pos_vals_" + str(each_threshold), "mean_neg_vals_" + str(each_threshold), "Num_of_CpG_islands_" + str(each_threshold)]

def get_length(current):
    length = int(current["End"].values[0] - current["Start"].values[0])
    return length

def extract_cg_count(chromosome, start, end, flank_length, genome_fasta):
    # Load the  reference genome
    # Extract the sequence for the given coordinates with additional flanking regions
    sequence_context = genome[chromosome].seq[(start - flank_length):(end + flank_length)]
    cg_count = sequence_context.upper().count('CG')
    return cg_count

###Slide window to calculate CpG density and if above threshold calculate mean methylation. 
def meth_at_CG_bins(Window_size,slide_distance,Obs_ExP_CpG_ratio,bw,genome,chromosome,start,stop):
    sequence_context = genome[chromosome].seq[(start):(stop)]
    Vals = bw.values(chromosome, start, stop)

    methylation_sum = 0
    window_start = 0
    window_end = window_start + Window_size

    list_pos_vals = []
    list_neg_vals = []
    #list_density = []
    list_CpG_status = [] # 0 is yes # 1 is no
    #list_GC_fraction = []
    #print(len(sequence_context))
    while window_end < len(sequence_context) + len(sequence_context)%slide_distance:
        window_sequence = sequence_context[window_start:window_end]
        #print(len(window_sequence))
        C_count_window = window_sequence.upper().count('C')
        G_count_window = window_sequence.upper().count('G')
        GC_content = ((C_count_window + G_count_window)/Window_size)*100
        CpG_count_window = window_sequence.upper().count('CG')
        if C_count_window * G_count_window >= 1:
            CpG_ratio = (CpG_count_window / (C_count_window * G_count_window))*Window_size
        else:
            CpG_ratio = 0
        #list_density.append(CpG_ratio)
        #list_GC_fraction.append(GC_content)
        
        #Count num of pos bins
        if CpG_ratio >= Obs_ExP_CpG_ratio and GC_content >= 50:
            list_CpG_status.append(1)
        else:
            list_CpG_status.append(0)
        
        slice_vals = Vals[window_start:window_end]
        if np.all(np.isnan(slice_vals)) == False:
            #list_vals.append(np.nanmean(slice_vals))
            if CpG_ratio >= Obs_ExP_CpG_ratio and GC_content >= 50:
                Meth_vals_window = np.nanmean(slice_vals)
                list_pos_vals.append(Meth_vals_window)
            else:
                Meth_vals_window = np.nanmean(slice_vals)
                list_neg_vals.append(Meth_vals_window)
        #else:
        #    list_vals.append(0.0)
        window_start = window_start + slide_distance 
        window_end = window_start + Window_size            
    
    if np.all(np.isnan(list_pos_vals)) == False:
        list_pos_vals = [x for x in list_pos_vals if not np.isnan(x)]
        mean_pos_vals =  np.mean(list_pos_vals)
        #std_pos_vals =  np.std(list_pos_vals)
    else:
        mean_pos_vals = float('nan')
        #std_pos_vals = float('nan')
    if np.all(np.isnan(list_neg_vals)) == False:
        list_neg_vals = [x for x in list_neg_vals if not np.isnan(x)]
        mean_neg_vals =  np.mean(list_neg_vals)
        #std_neg_vals =  np.std(list_pos_vals)
    else:
        mean_neg_vals = float('nan')
        #std_neg_vals = float('nan')
    
    #Count CpG islands
    # Convert the list to a numpy array
    arr = np.array(list_CpG_status)
    # Calculate the differences between consecutive elements
    diffs = np.diff(arr)
    # Count the number of times 1 is followed by -1 (indicating the end of a sequence of 1s)
    Num_of_CpG_islands = (diffs == -1).sum()
    return mean_pos_vals,mean_neg_vals,Num_of_CpG_islands#optional to save list_CpG_status ,list_density,Vals

def Get_metrics(gene_name):
    current = current_bed[current_bed["name"]==gene_name]
    chromosome = current["Chrom"].values[0]
    start = current["Start"].values[0]
    stop = current["End"].values[0]
    
    # calculate summary stats
    try:
        mean_meth = bw.stats(chromosome, int(start), int(stop),type="mean",exact=True)[0]
        std_meth = bw.stats(chromosome, int(start), int(stop),type="std",exact=True)[0]
    except RuntimeError:
        mean_meth = float('nan')
        std_meth = float('nan')

    # calculate dynamic stats
    try:
        Sliding_vals = []
        for each_threshold in CpG_thresholds_list:
            mean_pos_vals ,mean_neg_vals, Num_of_CpG_islands = meth_at_CG_bins(Window_size,slide_distance,each_threshold,bw,genome,chromosome,start,stop)    
            Sliding_vals = Sliding_vals + [mean_pos_vals,mean_neg_vals,Num_of_CpG_islands]

        # calculate ref stats
        Vals = bw.values(chromosome, int(start), int(stop))
        if len([x for x in Vals if str(x) != 'nan']) >= 1:
            Num_CpG = len([x for x in Vals if str(x) != 'nan'])
        else:
            Num_CpG = 0

        #num CG ref seq
        Num_ref_CpG = extract_cg_count(chromosome, int(start), int(stop), flank_length, genome_fasta)           
        length = get_length(current)

    except RuntimeError:
        Sliding_vals = []        
        for each_threshold in CpG_thresholds_list:
            Sliding_vals = Sliding_vals + [float('nan'),float('nan'),float('nan')]
        Num_CpG = float('nan')
        Num_ref_CpG = float('nan')
        length = float('nan')

    vals = [gene_name,Num_CpG,mean_meth,std_meth,Num_ref_CpG,length] + Sliding_vals

    
    return(vals)

#parallellize
p = Pool(processes=num_cpus)
Gene_list = current_bed["name"].to_list()

with p:
    results = p.map(Get_metrics,Gene_list)

merged_results = []
for i in range(len(results)):
    merged_results.append(results[i])
merged_results = pd.DataFrame(merged_results)
merged_results.columns = ["name","Num_CpG","mean_meth","std_meth","Num_ref_CpG","length"] + Sliding_vals_name 
merged_results.to_csv(out_dir+ "/" + site_name + "/"+out_name+ ".Meth_profile.tsv",sep = '\t',index = False)
print(in_file.split("/")[-1],"done")
