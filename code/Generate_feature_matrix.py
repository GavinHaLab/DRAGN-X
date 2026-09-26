#MA_25_194
#ml Python/3.9.6-GCCcore-11.2.0

import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample_name", type=str, required=True)
parser.add_argument("--path_to_results", type=str, required=True)
parser.add_argument("--path_to_TSS_file", type=str, required=True)
parser.add_argument("--path_to_save", type=str, required=True)
args = parser.parse_args()

sample_name = args.sample_name + ".Meth_profile.tsv"
path_to_results = args.path_to_results
path_to_TSS_file = args.path_to_TSS_file
path_to_save = args.path_to_save

def _load_sample_bed(bed_dir, sample_name):
    """Load and normalize a single BED-derived feature table."""
    bed_name = os.path.basename(bed_dir)

    df = pd.read_csv(os.path.join(bed_dir, sample_name), sep="\t")
    df = df.drop_duplicates(subset="name", keep="first")
    
    # Add sample
    df["sample"] = sample_name

    # Suffix all columns EXCEPT 'name' to keep features unique
    cols_to_suffix = [col for col in df.columns if col not in ['name', 'sample']]
    suffix_dict = {col: f"{col}_{bed_name}" for col in cols_to_suffix}
    df = df.rename(columns=suffix_dict)

    # Set 'name' as index for proper joining
    df = df.set_index('name')
    return df


def make_feat_matrix_2(
    sample_name,
    path_to_results,
    path_to_TSS_file,
    path_to_save,
):
    # --------------------------------------------------
    # Discover BED result directories
    # --------------------------------------------------
    bed_dirs = [
        os.path.join(path_to_results, d)
        for d in os.listdir(path_to_results)
        if not d.startswith(".")
    ]
       
    if not bed_dirs:
        raise ValueError("No BED result directories found.")

    # --------------------------------------------------
    # Load all BEDs and collect features
    # --------------------------------------------------
    dfs_to_merge = []

    for bed_dir in bed_dirs:
        df = _load_sample_bed(bed_dir, sample_name)
        dfs_to_merge.append(df)

    # --------------------------------------------------
    # Concatenate all BED features using 'name' as index
    # --------------------------------------------------
    merged_df = pd.concat(dfs_to_merge, axis=1, join='outer')
    
    # Reset index to make 'name' a column again
    merged_df = merged_df.reset_index()
    
    # Extract sample name (should be same across all, take from first column)
    sample_col = [col for col in merged_df.columns if col == 'sample']
    if sample_col:
        # Take the first 'sample' column and drop duplicates
        merged_df['sample'] = merged_df[sample_col[0]]
        # Drop all other 'sample' columns
        cols_to_drop = [col for col in merged_df.columns if col.startswith('sample') and col != 'sample']
        merged_df = merged_df.drop(columns=cols_to_drop)

    # --------------------------------------------------
    # Load and clean TSS BED
    # --------------------------------------------------
    tss_df = pd.read_csv(path_to_TSS_file, sep="\t", header = None)
    tss_df.columns = (
        ["Chrom", "Start", "End", "name", "score", "strand"]
        + tss_df.columns[6:].tolist()
    )
        
    # --------------------------------------------------
    # Merge TSS annotation on 'name'
    # --------------------------------------------------
    merged_df = merged_df.merge(tss_df, on="name", how="right")

    # --------------------------------------------------
    # Feature engineering
    # --------------------------------------------------
    merged_df["gene_size"] = merged_df["End"] - merged_df["Start"]
    merged_df["log_gene_size"] = np.log10(merged_df["gene_size"])

    #display(merged_df.head())
    #print(merged_df.columns.value_counts())
    
    # Drop all duplicate columns, keeping the first occurrence
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated(keep='first')]

    # Clean sample name
    merged_df["sample"] = merged_df["sample"].str.replace(
        ".Meth_profile.tsv", "", regex=False
    )

    
    # --------------------------------------------------
    # Reorder columns: put key columns first
    # --------------------------------------------------
    key_cols = ['sample', 'name', 'Chrom', 'Start', 'End', 'score', 'strand', 'gene_size', 'log_gene_size']
    other_cols = [col for col in merged_df.columns if col not in key_cols]
    merged_df = merged_df[key_cols + other_cols]

    # --------------------------------------------------
    # Save output
    # --------------------------------------------------
    merged_df.to_csv(path_to_save, index=False, sep='\t',compression='gzip')

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------
    print("Total rows:", len(merged_df))
    print("Genes:", merged_df["name"].nunique())
    print("Samples:", merged_df["sample"].nunique())
    print("Columns:", merged_df.shape[1])
    print("Saved to:", path_to_save)

#Run make_feat_matrix_2
print("Running Generate_feature_matrix.")
make_feat_matrix_2(sample_name, path_to_results, path_to_TSS_file, path_to_save)
