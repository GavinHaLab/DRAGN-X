#!/usr/bin/env python
# coding: utf-8

# In[63]:


from torch import nn
from torch import optim
import torch.nn.functional as F
"from torch.utils.data import DataLoader, TensorDataset"
import torch
from torch.utils.data import DataLoader, TensorDataset

import time
import pandas as pd
import numpy as np
from IPython.display import display
import random

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
import pickle

from sklearn.preprocessing import MinMaxScaler
"from joblib import dump, load  # Use joblib for model serialization"
from scipy import stats
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
import math
import argparse
from joblib import dump, load  # Use joblib for model serialization
from model_arch import Autoencoder


parser = argparse.ArgumentParser()
parser.add_argument("--Path_to_trained_model", type=str, required=True)
parser.add_argument("--feature_matrix", type=str, required=True)
parser.add_argument("--Methylation_scale", type=float, required=True)
parser.add_argument("--sample_sex", type=str, required=True)
parser.add_argument("--enhancement_factor", type=float, required=True)
parser.add_argument("--background_reference_Male", type=str, required=True)
parser.add_argument("--background_reference_Female", type=str, required=True)
parser.add_argument("--path_to_save", type=str, required=True)
args = parser.parse_args()


Path_to_trained_model = args.Path_to_trained_model
DRAGNX_data = args.feature_matrix
Methylation_scale = float(args.Methylation_scale)
sample_sex = args.sample_sex
enhancement_factor = float(args.enhancement_factor)
background_reference_Male =  args.background_reference_Male
background_reference_Female =  args.background_reference_Female
path_to_save = args.path_to_save


# In[16]:


##Load base model
Path_to_trained_model_BM = Path_to_trained_model + 'BaseModel/Meth2Exp_v19/'
file_name = Path_to_trained_model_BM + 'Meth2Exp_XGBc_v19.sav'
loaded_xgbc_model = pickle.load(open(file_name, "rb"))
feature_names_c = loaded_xgbc_model.get_booster().feature_names

##Load base model
file_name = Path_to_trained_model_BM + 'Meth2Exp_XGBc_Group_Neg_v19.sav'
loaded_xgbc_model_P_Neg_G_Neg = pickle.load(open(file_name, "rb"))
feature_names_P_Neg_G_Neg = loaded_xgbc_model_P_Neg_G_Neg.get_booster().feature_names
genes_xgbc_model_P_Neg_G_Neg = list(pd.read_csv(Path_to_trained_model_BM + 'Meth2Exp_XGBc_Group_Neg_v19_genes.csv')["name"])

##Load base model
file_name = Path_to_trained_model_BM + 'Meth2Exp_XGBc_Group_P_Neg_G_Pos_v19.sav'
loaded_xgbc_model_P_Neg_G_Pos = pickle.load(open(file_name, "rb"))
feature_names_P_Neg_G_Pos = loaded_xgbc_model_P_Neg_G_Pos.get_booster().feature_names
genes_xgbc_model_P_Neg_G_Pos = list(pd.read_csv(Path_to_trained_model_BM + 'Meth2Exp_XGBc_Group_P_Neg_G_Pos_v19_genes.csv')["name"])

##Load base model
file_name = Path_to_trained_model_BM + 'Meth2Exp_XGBc_Group_P_Pos_G_Neg_v19.sav'
loaded_xgbc_model_P_Pos_G_Neg = pickle.load(open(file_name, "rb"))
feature_names_P_Pos_G_Neg = loaded_xgbc_model_P_Pos_G_Neg.get_booster().feature_names
genes_xgbc_model_P_Pos_G_Neg = list(pd.read_csv(Path_to_trained_model_BM + 'Meth2Exp_XGBc_Group_P_Pos_G_Neg_v19_genes.csv')["name"])

##Load base model
file_name = Path_to_trained_model_BM + 'Meth2Exp_XGBc_Group_Pos_v19.sav'
loaded_xgbc_model_P_Pos_G_Pos = pickle.load(open(file_name, "rb"))
feature_names_P_Pos_G_Pos = loaded_xgbc_model_P_Pos_G_Pos.get_booster().feature_names
genes_xgbc_model_P_Pos_G_Pos = list(pd.read_csv(Path_to_trained_model_BM + 'Meth2Exp_XGBc_Group_Pos_v19_genes.csv')["name"])

##Load base model
file_name = Path_to_trained_model_BM + 'Meth2Exp_XGBrs_v19.sav'
loaded_xgbrs_model = pickle.load(open(file_name, "rb"))
feature_names_r = loaded_xgbrs_model.get_booster().feature_names

##Load base model
file_name = Path_to_trained_model_BM + 'Meth2Exp_XGBc_Group_Not_associated_v19.sav'
loaded_xgbc_model_Group_Not_associated = pickle.load(open(file_name, "rb"))
feature_names_Group_Not_associated = loaded_xgbc_model_Group_Not_associated.get_booster().feature_names
genes_xgbc_model_Group_Not_associated = list(pd.read_csv(Path_to_trained_model_BM + 'Meth2Exp_XGBc_Group_Not_associated_v19_genes.csv')["name"])


# In[102]:


def BaseModel_1(data):
    print("Applying BaseModels")
    ## Test ##

    ##Model 1c
    data["Model_1c_Prob"] = loaded_xgbc_model.predict_proba(data[feature_names_c])[:,1]
    #print("Model_1c done!")
        
    #Model GB Neg Pos
    data["Model_1c_Prob_P_Neg_G_Neg"] = loaded_xgbc_model_P_Neg_G_Neg.predict_proba(data[feature_names_P_Neg_G_Neg])[:,1]
    data["Model_1c_Prob_P_Neg_G_Pos"] = loaded_xgbc_model_P_Neg_G_Pos.predict_proba(data[feature_names_P_Neg_G_Pos])[:,1]
    data["Model_1c_Prob_P_Pos_G_Neg"] = loaded_xgbc_model_P_Pos_G_Neg.predict_proba(data[feature_names_P_Pos_G_Neg])[:,1]
    data["Model_1c_Prob_P_Pos_G_Pos"] = loaded_xgbc_model_P_Pos_G_Pos.predict_proba(data[feature_names_P_Pos_G_Pos])[:,1]
    data["Model_1c_Prob_Not_associated"] = loaded_xgbc_model_Group_Not_associated.predict_proba(data[feature_names_Group_Not_associated])[:,1]
        
    ##Model 1c2
    data["Model_1c2_Prob"] = data["Model_1c_Prob"]
    data.loc[data['name'].isin(genes_xgbc_model_P_Neg_G_Neg),"Model_1c2_Prob"] = data["Model_1c_Prob_P_Neg_G_Neg"]
    data.loc[data['name'].isin(genes_xgbc_model_P_Neg_G_Pos),"Model_1c2_Prob"] = data["Model_1c_Prob_P_Neg_G_Pos"]
    data.loc[data['name'].isin(genes_xgbc_model_P_Pos_G_Neg),"Model_1c2_Prob"] = data["Model_1c_Prob_P_Pos_G_Neg"]
    data.loc[data['name'].isin(genes_xgbc_model_P_Pos_G_Pos),"Model_1c2_Prob"] = data["Model_1c_Prob_P_Pos_G_Pos"]
    data.loc[data['name'].isin(genes_xgbc_model_Group_Not_associated),"Model_1c2_Prob"] = data["Model_1c_Prob_Not_associated"]

    #print("Model_1c2 done!")
    
    ##Model 1r all
    data["Model_1r_Prob"] = loaded_xgbrs_model.predict(data[feature_names_r])
    #print("Model_1r done!")
    
    print('BaseModel predictions made')
    return data


# In[99]:


##Apply model_2
# Use CPU
# define a function to apply autoencoder to a dataset
def get_autoendoder_predictions(df,select_genes,Highly_on_genes,on_genes,off_genes,grey_genes,predictions_in, predictions_out):
    
    Feature_matrix = pd.pivot_table(df, values=predictions_in, index=['sample'],columns=['name'])
    #print('isna',Feature_matrix.isna().sum().sum())
    Feature_matrix = Feature_matrix.fillna(0.5)
    #print(Feature_matrix.shape)
    # Predict using autoencoder
    test_data_DF = Feature_matrix[select_genes]
    #print(test_data_DF.shape)

    # get test data
    test_names = test_data_DF.index
    test_data = test_data_DF.values

    # make test data into a tensor
    test_data = torch.FloatTensor(test_data)

    # make test data into a dataloader
    test_data = DataLoader(test_data)
    
    # predict test data
    test_pred = []

    # set model to evaluation mode
    model.eval()

    # loop over test data
    for data in test_data:
        # get the inputs
        inputs = data
        # Separate the sets of features
        set_1_index = [test_data_DF.columns.get_loc(i) for i in Highly_on_genes]        
        set_2_index = [test_data_DF.columns.get_loc(i) for i in on_genes]        
        set_3_index = [test_data_DF.columns.get_loc(i) for i in off_genes]        
        set_4_index = [test_data_DF.columns.get_loc(i) for i in grey_genes]     

        set_1_input = inputs[:,set_1_index].to(device)
        set_2_input = inputs[:,set_2_index].to(device)
        set_3_input = inputs[:,set_3_index].to(device) 
        set_4_input = inputs[:,set_4_index].to(device) 

        # forward
        outputs = model(set_1_input,set_2_input,set_3_input,set_4_input)
        # add outputs to test_pred
        test_pred.append(outputs.detach().cpu().numpy())

    # Make predicted data into a dataframe
    test_pred = np.concatenate(test_pred)
    test_pred = pd.DataFrame(test_pred)
    # add row and column names to predicted data
    test_pred.index = test_names
    test_pred.columns = select_genes
    #print(test_pred.shape)
    #display(test_pred.head())

    # Reformat Model_output into three columns sample, gene and pred
    Model_output = test_pred.stack().reset_index()
    Model_output.columns = ["sample", "name", predictions_out]
    # Make a column called sample_gene and merge with current_DF_test
    Model_output["sample_gene"] = Model_output["sample"] + "_" + Model_output["name"]
    #print(len(Model_output['sample'].unique()))
    #print(len(Model_output['name'].unique()))
    
    Model_output = Model_output.drop(columns = ["sample","name"])

    
    df = pd.merge(df,Model_output,on = "sample_gene",how = "inner")

    del test_data
    return df


# In[100]:


## Train Meth2LS
#  Define a function to get the last encoder layer latent space for a dataset
def get_latent_space(df,select_genes,Highly_on_genes,on_genes,off_genes,grey_genes,test_names,predict_in):
	
	test_data_DF = pd.pivot_table(df, values=predict_in, index=['sample'],columns=['name'])
#	print('isna',test_data_DF.isna().sum().sum())
	test_data_DF = test_data_DF.fillna(0.5)
	# Predict using autoencoder
	test_data_DF = test_data_DF[select_genes]
#	print(test_data_DF.shape)

	# get test data
	test_names = test_data_DF.index
	test_data = test_data_DF.values

	# make test data into a tensor
	test_data = torch.FloatTensor(test_data)

	# make test data into a dataloader
	test_data = DataLoader(test_data)
	
	# predict test data
	test_pred = []

	# set model to evaluation mode
	model.eval()

	# loop over test data
	for data in test_data:
		# get the inputs
		inputs = data.to(device)
        
		# Separate the sets of features
		set_1_index = [test_data_DF.columns.get_loc(i) for i in Highly_on_genes]
		set_2_index = [test_data_DF.columns.get_loc(i) for i in on_genes]
		set_3_index = [test_data_DF.columns.get_loc(i) for i in off_genes]
		set_4_index = [test_data_DF.columns.get_loc(i) for i in grey_genes]

		set_1_input = inputs[:,set_1_index].to(device)
		set_2_input = inputs[:,set_2_index].to(device)
		set_3_input = inputs[:,set_3_index].to(device)
		set_4_input = inputs[:,set_4_index].to(device)

		# forward
		outputs_1 = model.encoder_1(set_1_input)
		outputs_2 = model.encoder_2(set_2_input)        
		outputs_3 = model.encoder_3(set_3_input)   
		outputs_4 = model.encoder_4(set_4_input)        
		outputs =   torch.cat((outputs_1, outputs_2, outputs_3,outputs_4), dim=1)
        
		# add outputs to test_pred
		test_pred.append(outputs.detach().cpu().numpy())

	# Make predicted data into a dataframe
	test_pred = np.concatenate(test_pred)
	test_pred = pd.DataFrame(test_pred)

	# Add row name
	test_pred.index = test_names
	test_pred = test_pred.T
#	print(test_pred.shape)
	return test_pred


# In[57]:


##Fine tune Encoder
import copy
class EncoderOnly(nn.Module):
    def __init__(self, trained_model):
        super().__init__()

        self.encoder_1 = copy.deepcopy(trained_model.encoder_1)
        self.encoder_2 = copy.deepcopy(trained_model.encoder_2)
        self.encoder_3 = copy.deepcopy(trained_model.encoder_3)
        self.encoder_4 = copy.deepcopy(trained_model.encoder_4)

        # ensure trainable
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x1, x2, x3, x4):
        z1 = self.encoder_1(x1)
        z2 = self.encoder_2(x2)
        z3 = self.encoder_3(x3)
        z4 = self.encoder_4(x4)

        return torch.cat([z1, z2, z3, z4], dim=1)


# In[58]:


##Apply link model - Meth2LS2ExP
def Test_Meth2ExpLS_NN(model_2,test_names,stacked_features):
    # Predict using model
    test_pred = []

    # Set model to evaluation mode
    model_2.eval()

    # Loop over test data
    for each_sample in test_names:
        # Get the inputs
        inputs = torch.FloatTensor(stacked_features.loc[each_sample].values).to(device)
        
        # Forward
        outputs = model_2(inputs) 

        # Add outputs to test_pred
        test_pred.append(outputs.detach().cpu().numpy())

    # Add row and column names to predicted data
    test_pred = pd.DataFrame(test_pred)
    test_pred.index = test_names
    return(test_pred)


# In[59]:


def Test_FineTune_Encoder(model_2, test_names, stacked_features):

    model_2.eval()

    test_pred = []

    # Precompute feature indices (same logic as training)
    set_1_index = [stacked_features.columns.get_loc(i) for i in Highly_on_genes]
    set_2_index = [stacked_features.columns.get_loc(i) for i in on_genes]
    set_3_index = [stacked_features.columns.get_loc(i) for i in off_genes]
    set_4_index = [stacked_features.columns.get_loc(i) for i in grey_genes]

    with torch.no_grad():

        for each_sample in test_names:

            # single sample input
            sample = stacked_features.loc[each_sample].values

            x1 = torch.FloatTensor(sample[set_1_index]).unsqueeze(0).to(device)
            x2 = torch.FloatTensor(sample[set_2_index]).unsqueeze(0).to(device)
            x3 = torch.FloatTensor(sample[set_3_index]).unsqueeze(0).to(device)
            x4 = torch.FloatTensor(sample[set_4_index]).unsqueeze(0).to(device)

            outputs = model_2(x1, x2, x3, x4)

            test_pred.append(outputs.squeeze(0).cpu().numpy())

    test_pred = pd.DataFrame(test_pred)
    test_pred.index = test_names

    return test_pred


# In[101]:


###Test LS model
def Model_Meth2LS(data,Input_Model_1,Trained_Meth2LS):
    ## Model 3
    stacked_features = pd.pivot_table(data, values=Input_Model_1, index=['sample'],columns=['name'])
    stacked_features = stacked_features[input_col_order]
    select_LS_Samples = list(stacked_features.index)
    #Predicted_LS_DF = Test_Meth2ExpLS_NN(Trained_Meth2LS,select_LS_Samples,stacked_features)
    Predicted_LS_DF = Test_FineTune_Encoder(Trained_Meth2LS,select_LS_Samples,stacked_features)
    
    ##Retransform
    #df_transposed_rescaled = loaded_LS_scaler.inverse_transform(Predicted_LS_DF)
    #Predicted_LS_DF = pd.DataFrame(df_transposed_rescaled, columns=Predicted_LS_DF.columns, index=Predicted_LS_DF.index)#
    
    return Predicted_LS_DF

##Retransform model
def Reverse_transform_B2C(df,predname,new_pred_name):
    # Reverse transform to the original form
    loaded_scaler = load(Path_to_trained_model + 'DRAGNX-AE/V18_1M/scaler_model_TPM_DF_v2.joblib')
    
    #loaded_scaler = load('/fh/fast/haffner_m/user/madil/project/DRAGNX/DRAGNX-AE/Exp2Exp_v9/GTEx/Fine_tune/TCGA/Fine_tune/MET500/scaler_model_TPM_DF_v2.joblib')
    #loaded_scaler = load('/fh/fast/haffner_m/user/madil/project/DRAGNX/DRAGNX-AE/Exp2Exp_v1/GTEx/Fine_tune/TCGA/scaler_model_GTEx_v1.joblib')
    #loaded_scaler = load('/fh/fast/haffner_m/collab_space/projects/DRAGNX/model_data/scaler_model2_TCGA.joblib')

    #loaded_scaler = load('scaler_model_Advance_cancer_v1.joblib')
    
    Current_DF = pd.pivot_table(df, values=predname, index=['sample'],columns=['name'])

    # Load the scaler model
    transform_feats = list(loaded_scaler.get_feature_names_out())
    #print(transform_feats)
    Current_DF = Current_DF[transform_feats]
    #print(Current_DF.shape)
    #display(Current_DF.iloc[:5,:5])

    df_transposed_rescaled = loaded_scaler.inverse_transform(Current_DF)
    df_transposed_rescaled = pd.DataFrame(df_transposed_rescaled, columns=Current_DF.columns, index=Current_DF.index)#
    #print(df_transposed_rescaled.shape)
    #display(df_transposed_rescaled.iloc[:5,:5])

    df_transposed_rescaled = df_transposed_rescaled.reset_index().melt(id_vars='sample', var_name='name', value_name=new_pred_name)
    df_transposed_rescaled["sample_gene"] = df_transposed_rescaled["sample"] + "_" + df_transposed_rescaled["name"]

    df = pd.merge(df,df_transposed_rescaled[['sample_gene',new_pred_name]],on = 'sample_gene',how = 'inner')
    return df


##Apply Decoder
def Model3(data,Input_Model_1,Trained_Meth2LS,Output_name):
    ## Model 3
    stacked_features = pd.pivot_table(data, values=Input_Model_1, index=['sample'],columns=['name'])
    stacked_features = stacked_features[input_col_order]
    select_LS_Samples = list(stacked_features.index)
    #Predicted_LS_DF = Test_Meth2ExpLS_NN(Trained_Meth2LS,select_LS_Samples,stacked_features)
    Predicted_LS_DF = Test_FineTune_Encoder(Trained_Meth2LS,select_LS_Samples,stacked_features)

    #print("LS created...")
    # Apply decoder to predicted latent space
    test_decoded = torch.FloatTensor(Predicted_LS_DF.values).to(device)
    test_decoded = model.decoder(test_decoded)
    test_decoded = test_decoded.detach().cpu().numpy()
    test_decoded = pd.DataFrame(test_decoded)
    test_decoded.index = select_LS_Samples
    test_decoded.columns = input_col_order
    test_decoded = test_decoded.reset_index().melt(id_vars='index', var_name='name', value_name=Input_Model_1)
    test_decoded.columns = ["sample","name",Output_name]
    test_decoded["sample_gene"] = test_decoded["sample"] + "_" + test_decoded["name"]
    data = pd.merge(data,test_decoded[["sample_gene",Output_name]],on = "sample_gene",how = "inner")
    #print("Decoded!")
    
    Output_name_scaled = Output_name + "_scaled"
    data = Reverse_transform_B2C(data,Output_name,Output_name_scaled)
    
    #print("Scaled!")
    return data


# In[ ]:


##Load Healthy data
Healthy_male_ref = pd.read_csv(background_reference_Male,compression = 'gzip')
Healthy_male_ref = Healthy_male_ref[Healthy_male_ref["Chrom"]!="chrUn_gl000228"]
drop_index = Healthy_male_ref[(Healthy_male_ref['name']=='SIGLEC5')&(Healthy_male_ref['Start']==52115180)].index
Healthy_male_ref = Healthy_male_ref.drop(drop_index)


# In[ ]:


##Load Healthy data
Healthy_female_ref = pd.read_csv(background_reference_Female,compression = 'gzip')
Healthy_female_ref = Healthy_female_ref[Healthy_female_ref["Chrom"]!="chrUn_gl000228"]
drop_index = Healthy_female_ref[(Healthy_female_ref['name']=='SIGLEC5')&(Healthy_female_ref['Start']==52115180)].index
Healthy_female_ref = Healthy_female_ref.drop(drop_index)


# In[36]:


##Load Tissue reference
Ref_NI_ref_q01 = pd.read_csv(Path_to_trained_model + 'Parameters/NI_ref_df_q01.csv',index_col = 'name')
Ref_NI_ref_q99 = pd.read_csv(Path_to_trained_model + 'Parameters/NI_ref_df_q99.csv',index_col = 'name')


# In[23]:


##IMFx Estimator
#Load model
file_name = Path_to_trained_model + 'ImmuneFractionEstimator/IMFx_deep_HighLR_lowReg_v4.sav'
loaded_IMFx_xgbr_model = pickle.load(open(file_name, "rb"))
print("Model Loaded")

def split_string_list(strings):
    first_parts = []
    second_parts = []
    
    for item in strings:
        split_index = item.find('_mean')
        if split_index != -1:
            first_parts.append(item[:split_index])
            second_parts.append(item[split_index+1:])  # include 'm' by offsetting +1
        else:
            first_parts.append(item)
            second_parts.append('')
    
    first_parts = list(set(first_parts))
    second_parts = list(set(second_parts))
    
    return first_parts, second_parts


def IMFx_Estimator(data,loaded_IMFx_xgbr_model):

    #Get feature names
    IMFx_feature_names_r = loaded_IMFx_xgbr_model.get_booster().feature_names
    IMFx_feature_genes, IMFx_features = split_string_list(IMFx_feature_names_r)

    Feat_data = data.copy()
    Feat_data = Feat_data[Feat_data["name"].isin(IMFx_feature_genes)]
    Feat_data = Feat_data[['sample','name']+IMFx_features]
    Feat_data = Feat_data.fillna(0)

    # Step 1: Melt to long format
    df_melted = Feat_data.melt(id_vars=['sample', 'name'], var_name='feature', value_name='value')
    del Feat_data

    # Step 2: Create new column: gene_feature
    df_melted['gene_feature'] = df_melted['name'] + '_' + df_melted['feature']

    # Step 3: Pivot using pivot_table (in case of accidental duplicates)
    df_wide = df_melted.pivot_table(
        index='sample',
        columns='gene_feature',
        values='value',
        #aggfunc='mean'  # Or 'first', 'sum', etc.
    )
    del df_melted

    # Replace 'NA' with np.NaN
    df_wide = df_wide.replace(-1, np.nan)

    # Step 4: Apply model
    df_wide["IMFx"] = loaded_IMFx_xgbr_model.predict(df_wide[IMFx_feature_names_r])

    df_wide = df_wide[['IMFx']]
    #df_wide['IMFx'] = 1 - df_wide['IMFx']
    df_wide['IMFx'] = df_wide['IMFx'].clip(0,1)
    df_wide = df_wide.reset_index()

    data = pd.merge(data, df_wide, on = 'sample',how = 'inner')   
    #del df_melted
    #print('IMFx calculated')
    #Test_tmp.head()

    return(data)


# In[24]:


def Robust6_z_DF(data,Blood_DF,Ref_NI_ref_q01,Ref_NI_ref_q99, enhance_factor, enhance_feats):
    
    if enhance_factor > 0:
        # Save a copy
        OG_data = data.copy()

        # Remove old index
        OG_data.reset_index(drop=True, inplace=True)

        # Copy dataframes
        ref_df = Blood_DF.copy()
        test_df = OG_data.copy()

        test_df.index = test_df['sample_gene']

        # Ensure gene_name is the index for grouping
        ref_df.set_index("name", inplace=True)
        test_df.set_index("name", inplace=True)
        
        # Get immune fraction
        IMFx_df = test_df[['IMFx']].reindex(test_df.index)
        sample_gene_index = list(test_df['sample_gene'])

        # Get gene_size
        gene_size_df = test_df[['gene_size']].reindex(test_df.index)
        
        # Ensure same columns
        ref_df = ref_df[enhance_feats]
        Ref_NI_ref_q01 = Ref_NI_ref_q01[enhance_feats]
        Ref_NI_ref_q99 = Ref_NI_ref_q99[enhance_feats]    
        test_df = test_df[enhance_feats]

        # Compute reference statistics per gene
        ref_median = ref_df.groupby(level=0).median()
        #ref_mad = ref_df.groupby(level=0).mad().replace(0, np.nan)  # Replace 0 MAD with NaN to avoid division errors

        # Align reference stats with test_df (broadcasting per gene)
        ref_median = ref_median.reindex(test_df.index)
        #ref_mad = ref_mad.reindex(test_df.index)

        Ref_NI_ref_q01 = Ref_NI_ref_q01.reindex(test_df.index)
        Ref_NI_ref_q99 = Ref_NI_ref_q99.reindex(test_df.index)

        # calculate delta
        delta = test_df - ref_median

        ##Keep large diff large
        ##keep small diff small
#         large_diff_cutoff = 75
        small_diff_cutoff = 5
        delta[(delta>-small_diff_cutoff)&(delta<small_diff_cutoff)] = 0
#         delta[(delta<-large_diff_cutoff)] = -large_diff_cutoff
#         delta[(delta>large_diff_cutoff)] = large_diff_cutoff        
        
        enhance_feats_TSS = [ x for x in enhance_feats if "TSS"  in x ] +  [ x for x in enhance_feats if "Promoter"  in x ]
        enhance_feats_TSS = [ x for x in enhance_feats_TSS if "_500"  in x ]

        enhance_feats_GB = [ x for x in enhance_feats if "GB"  in x ] +  [ x for x in enhance_feats if "Gene"  in x ]
        enhance_feats_GB = [ x for x in enhance_feats_GB if "Seg"  in x ]
        
        TSS_small_diff_cutoff = 10
        mask = (delta[enhance_feats_TSS] > -TSS_small_diff_cutoff) & (delta[enhance_feats_TSS] < TSS_small_diff_cutoff)
        delta.loc[:, enhance_feats_TSS] = delta.loc[:, enhance_feats_TSS].mask(mask, 0)

        GB_small_diff_cutoff = 10
        Small_gene_rows = gene_size_df[gene_size_df["gene_size"] < 10000].index

        mask = (
            (delta.loc[Small_gene_rows, enhance_feats_GB] > -GB_small_diff_cutoff) &
            (delta.loc[Small_gene_rows, enhance_feats_GB] <  GB_small_diff_cutoff))

        delta.loc[Small_gene_rows, enhance_feats_GB] = (
            delta.loc[Small_gene_rows, enhance_feats_GB].mask(mask, 0))
        
        
        # scale the delta by IMFx and use enhancement factor to calibrate scaling        
        IMFx_df.loc[IMFx_df['IMFx'] > 0.8, 'IMFx'] = 1
        
        exponential_factor = 1
        multiplier = (1-IMFx_df['IMFx']**exponential_factor) * enhance_factor 
        
        delta = delta.multiply(multiplier, axis=0)

        # Add delta back to test 
        robust_z_df = test_df + delta

        # Clip to remove outlier values
        robust_z_df = robust_z_df.clip(lower=Ref_NI_ref_q01, upper=Ref_NI_ref_q99)

        # Fill back original na using test_df
        robust_z_df = robust_z_df.where(~(test_df.isna()))
        robust_z_df = robust_z_df.fillna(test_df)

        #robust_z_df.index = sample_gene_index
        robust_z_df.reset_index(drop=False, inplace=True)
        robust_z_df['sample_gene'] = sample_gene_index

        robust_z_df = pd.merge(robust_z_df,OG_data[list(set(OG_data.columns)-set(robust_z_df.columns))+['sample_gene']],on = 'sample_gene',how = 'inner')
        robust_z_df = robust_z_df[OG_data.columns]
        #print("Signal Enhanced")
    else:
        robust_z_df = data
    return(robust_z_df)


# In[46]:


Highly_on_genes = pd.read_csv(Path_to_trained_model + 'Parameters/highly_on_genes_v4.csv')
Highly_on_genes = list(Highly_on_genes['0'])
print('Highly_on_genes',len(Highly_on_genes))

on_genes = pd.read_csv(Path_to_trained_model + 'Parameters/on_genes_v4.csv')
on_genes = list(on_genes['0'])
print('on_genes',len(on_genes))

off_genes = pd.read_csv(Path_to_trained_model + 'Parameters/off_genes_v4.csv')
off_genes = list(off_genes['0'])
print('off_genes',len(off_genes))

grey_genes = pd.read_csv(Path_to_trained_model + 'Parameters/grey_genes_v4.csv')
grey_genes = list(grey_genes['0'])
print('grey_genes',len(grey_genes))


# In[25]:


##Load AE model genes
gene_order_DF = pd.read_csv(Path_to_trained_model + "DRAGNX-AE/V18_1M/model_gene_order.txt",header = None)
input_col_order = gene_order_DF[0].to_list()
print(len(input_col_order))


##Load AE (weights-only: architecture lives in code/model_arch.py, not in Trained_Models/)
device = torch.device("cpu")
torch.cuda.empty_cache()
ae_checkpoint = torch.load(Path_to_trained_model + "DRAGNX-AE/V18_1M/5000_B2C.pth",
                            weights_only=True, map_location="cpu")
ae_config = ae_checkpoint["config"]
model = Autoencoder(ae_config["input_dim"], ae_config["s1"], ae_config["s2"],
                     ae_config["s3"], ae_config["s4"], ae_config["dropout"])
model.load_state_dict(ae_checkpoint["state_dict"])
model.eval()


# In[ ]:


##Output model name for both Continuour and Binary
Meth2LS_c2_name = 'AE_Exp2Exp_v18_Meth2Exp_XGBc2_v19_Meth2LS_v2'

##Path to Meth2LS
Meth2LS_path = Path_to_trained_model + 'Meth2LS/v18_1M/'


# In[97]:


##Start
print("Initiating DRAGN-X workflow")

DRAGNX_data = pd.read_csv(DRAGNX_data, sep='\t',compression = 'gzip')
drop_index = DRAGNX_data[(DRAGNX_data['name']=='SIGLEC5')&(DRAGNX_data['Start']==52115180)].index
DRAGNX_data = DRAGNX_data.drop(drop_index)
DRAGNX_data["sample_gene"] = DRAGNX_data["sample"] + "_" + DRAGNX_data["name"]

# Adjust methylation scale
mean_meth_columns = [col for col in DRAGNX_data.columns if "mean_" in col]

if Methylation_scale == 1:
    DRAGNX_data.loc[:, mean_meth_columns] *= 100

elif Methylation_scale == 100:
    # Already on 0–100 scale; do nothing
    pass

else:
    raise ValueError(
        f"Methylation_scale must be 1 or 100, "
        f"but got {Methylation_scale}"
    )

print("MethProfiler Data Loaded")

##Calculate IMFx 
DRAGNX_data = IMFx_Estimator(DRAGNX_data,loaded_IMFx_xgbr_model)

##Signal Enhanced 
DRAGNX_data["sample_gene"] = DRAGNX_data["sample"] + "_" + DRAGNX_data["name"]

enhance_feats = [ x for x in feature_names_c if "mean"  in x ]
if sample_sex == "male":
    print("Sex is " + sample_sex)
    DRAGNX_data = Robust6_z_DF(DRAGNX_data, Healthy_male_ref, Ref_NI_ref_q01,Ref_NI_ref_q99,enhancement_factor, enhance_feats)
    print('Signal Enhanced')
elif sample_sex == "female":
    print("Sex is " + sample_sex)
    DRAGNX_data = Robust6_z_DF(DRAGNX_data, Healthy_female_ref, Ref_NI_ref_q01,Ref_NI_ref_q99,enhancement_factor, enhance_feats)
    print('Signal Enhanced')
else:
    print("Sex (male/female) not defined")
    DRAGNX_data = DRAGNX_data.copy()
    print('Signal Not Enhanced')

##Apply Base Mode 
DRAGNX_data =  BaseModel_1(DRAGNX_data.reset_index(drop = True))
DRAGNX_data = Reverse_transform_B2C(DRAGNX_data,'Model_1c2_Prob','Model_1c2_Prob' + '_scaled')
DRAGNX_data = Reverse_transform_B2C(DRAGNX_data,'Model_1r_Prob','Model_1r_Prob' + '_scaled')

##Apply Model 2&3
print('Applying DRAGN-X model')
Fine_Tune_levels = [0,100,200,500,1000]
for each_epoc in Fine_Tune_levels:
    pred = "Model_1c2_Prob"
    select_Meth2LS_name = Meth2LS_c2_name
    Epoc_type = "_" + str(each_epoc)
    device = torch.device("cpu")
    Meth2LS = torch.load(Meth2LS_path + select_Meth2LS_name + Epoc_type + '.pth',map_location=torch.device(device), weights_only=False).to(device)
    DRAGNX_data = get_autoendoder_predictions(DRAGNX_data,input_col_order,Highly_on_genes,on_genes,off_genes,grey_genes,pred,pred.replace("1","2") + Epoc_type)
    DRAGNX_data = Reverse_transform_B2C(DRAGNX_data,pred.replace("1","2") + Epoc_type,pred.replace("1","2") + Epoc_type + '_scaled')
    DRAGNX_data = Model3(DRAGNX_data,pred,Meth2LS,pred.replace("1","3") + Epoc_type)
        
##Save data
DRAGNX_data.to_csv(path_to_save,index = False,compression = 'gzip')
print("DRAGN-X applied successfully!")
