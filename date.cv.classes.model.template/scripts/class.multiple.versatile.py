#!/usr/bin/env python

# __author__ = Firas Said Midani
# __e-mail__ = fsm3@duke.edu
# __update__ = 2015.12.07
# __version_ = 1.0

#########################################################
# DESCRIPTION
#########################################################
# Given a parameter file that calls the following
#  - feature_options
#  - hold in flags
#  - hold out flags
#  - outcomes dict
#  - frequency cutoff
#  - num_features
#  - coarse and fine steps
#  - whether to normalize
#  - whether to include bacterial and/or clinical features
#  - whether to pickle some of the final model output
#  - internal cross-validation number of folds
#  - cross-validation scheme
#  - classification method and parameterization
# 
# Do the following:
#  - Perform either multiple iterations of a classifier, with true labels or shuffled labels. 
# 
# Designed to run on a HPC (e.g. SLURM). It will create a job array with each corresponding
#  to a single classification (with selected cross-validation scheme, classifier, and 
#  optional parameter grid search). See class.single.versatile.rfe.py for more details. 

#########################################################
# INPUT 
#########################################################

# params             = sys.argv[1]; parameter file
# txt_featues_df     = sys.argv[2]; path to bacterial features matrix file
# txt_taxa_level_map = sys.argv[3]; path to features taxonomic level map
# txt_otu_taxa_map   = sys.argv[4]; path to OTU-->taxonomy map
# txt_mapping        = sys.argv[5]; path to samples mapping file
# txt_clinical_df    = sys.argv[6; path to clinical features file
# filepath           = sys.argv[7]; path to store reulsts
# simname            = sys.argv[8; simulation name (used for naming files)
# shuffle            = sys.argv[9]; 1="shuffle labels", 0="don't shuffle labels"
# numperm            = int(sys.argv[10]); "number of permutations/iterations
# pickle_model       = int(sys.argv[11]); 1="pickle select model output", 0="don't pickle anything"

##########################################################
# CREATES/MODIFIES/REMOVES
##########################################################

# CREATES
#  * DIR SLURM.LOG 
#  * TXT summary of this main job
#  * TXT (OUT) of shell output for this main job
#  * TXT (ERR) of shell errors for this main job
#  * TXT (SLURM) shell commands for submitting job array
#  * TXT of feature matrix
#
#  times number of iterations (jobs submitted by this script) 
#  * TXT of labels for each iteration 
#  * TXT (OUT) of shell output for each iteration 
#  * TXT (ERR) of shell errors for each iteration 
#  * TXT (OUT) file on directory where this main job is submitted (not sure why duplicate)
#  * TXT (ERR) file on directory where this main job is submitted (not sure why duplicate)
#
# REMOVES
# 
# KEY
# ~~ these are files created and removed by this script

##########################################################
# dependencies
##########################################################

#  cholera_classifie_libb: e.g. class.library.py
#  SLURM: designed for usage on a HPC.
 
##########################################################
# initialization 				   
##########################################################
					
import sys, imp, os

#####################################
# in-house tools
#####################################

pypath = os.path.dirname(os.path.realpath(sys.argv[0]));
foo    = imp.load_source('classification_library',pypath+'/class.library.py')
from classification_library import *

##########################################################
# seed random generator				   
##########################################################

seedint = random.randint(0, sys.maxint);
random.seed(seedint)

##########################################################
# input handling 				   
##########################################################

params             = sys.argv[1]; print 'params\t',params 
txt_features_df    = sys.argv[2]; print 'txt_features_df\t',txt_features_df 
txt_taxa_level_map = sys.argv[3]; print 'txt_taxa_level_map\t',txt_taxa_level_map
txt_otu_taxa_map   = sys.argv[4]; print 'txt_otu_taxa_map\t',txt_otu_taxa_map
txt_mapping        = sys.argv[5]; print 'txt_mapping\t',txt_mapping 
txt_clinical_df    = sys.argv[6]; print 'txt_clinical\t',txt_clinical_df
filepath           = sys.argv[7]; print 'filepath\t',filepath
simname            = sys.argv[8]; print 'simname\t',simname
shuffle            = int(sys.argv[9]); print 'shuffle\t',shuffle
numperm            = int(sys.argv[10]); print 'numperm\t',numperm
pickle_model       = int(sys.argv[11]); print 'pickle_model\t', pickle_model

foo = imp.load_source('model_parameters',params)
from model_parameters import *

# INITIALIZE JOB ARRAY REPOSITORY
if not os.path.isdir(filepath+'/slurm.log'):
    os.system('mkdir '+filepath+'/slurm.log');
 
##########################################################
# classifier input and parameters			   
##########################################################

#######################################################################
##Read feature matrix and feature meta-data
#######################################################################
features_df    = pd.read_csv(txt_features_df,sep='\t',header=0,index_col=0);
taxa_level_map = pd.read_csv(txt_taxa_level_map,sep='\t',header=0,index_col=0);
otu_taxa_map   = pd.read_csv(txt_otu_taxa_map,sep='\t',header=0,index_col=0);

#######################################################################
##Read sample labels.
#######################################################################
mapping           = pd.read_csv(txt_mapping,sep='\t',header=0,index_col=0)
mapping['family'] = [int(idx.split('.')[1]) for idx in mapping.index]

#######################################################################
##Define training data
#######################################################################
# only select samples that meet the holdout criteria
subset_samples    = subset_data([holdin_flag1,holdin_flag2],mapping,features_df)
# only select features that correspond to a certain resolution (e.g. family)
subset_features = subset_data([feature_options],taxa_level_map,features_df.transpose())
# create labels array and feature matrix
y_holdin_df       = mapping.loc[subset_samples.index,'color'].replace(outcomes_dict).astype(float)
x_holdin_df       = features_df.loc[subset_samples.index,subset_features.index].astype(float).fillna(0); 

#######################################################################
##Define validation data
#######################################################################
# only select samples that meet the holdout criteria
subset_samples   = subset_data([holdout_flag1,holdout_flag2],mapping,features_df)
# only select features that correspond to a certain resolution (e.g. family)
subset_features = subset_data([feature_options],taxa_level_map,features_df.transpose())
# create labels array and feature matrix
y_holdout_df       = mapping.loc[subset_samples.index,'color'].replace(outcomes_dict).astype(float)
x_holdout_df       = features_df.loc[subset_samples.index,subset_features.index].astype(float).fillna(0); 

#######################################################################
##Combine training and validation data
#######################################################################

x_all = pd.concat([x_holdin_df,x_holdout_df])
y_all = pd.concat([y_holdin_df,y_holdout_df])
x_all.shape,y_all.shape

#######################################################################
##SAVE FEATURE MATRIX (before filtering for less frequent features
#######################################################################

# SAVE FEATURE MATRIX (should be same for all jobs in array)
txt_x_holdin_df       = filepath+'/slurm.log/x_holdin_df.txt';
txt_x_holdout_df      = filepath+'/slurm.log/x_holdout_df.txt';
txt_x_all             = filepath+'/slurm.log/x_all.txt';

x_holdin_df.to_csv(txt_x_holdin_df,sep='\t',header=True,index_col=True);
x_holdout_df.to_csv(txt_x_holdout_df,sep='\t',header=True,index_col=True);
x_all.to_csv(txt_x_all,sep='\t',header=True,index_col=True);

#######################################################################
##Filter data based on frequency of presence of each feature across model samples
#######################################################################

bfd = pd.DataFrame(binarize(x_holdin_df),index=x_holdin_df.index,columns=x_holdin_df.keys())
dense_features = bfd.keys()[np.where(bfd.apply(np.sum)>=np.ceil(frequency_cutoff*x_holdin_df.shape[0]))[0]]

x_holdin_df    = x_holdin_df.loc[:,dense_features]
x_holdout_df   = x_holdout_df.loc[:,dense_features]

##########################################################
# distribute jobs input		   
##########################################################

# SAVE FEATURE MATRIX (should be same for all jobs in array)
txt_x_holdin_df       = filepath+'/slurm.log/x_holdin_df_dense.txt';
txt_x_holdout_df      = filepath+'/slurm.log/x_holdout_df_dense.txt';
txt_x_all             = filepath+'/slurm.log/x_all_dense.txt';
txt_y_holdin_df       = filepath+'/slurm.log/y_holdin_df.txt';
txt_y_holdout_df      = filepath+'/slurm.log/y_holdout_df.txt';
txt_y_all             = filepath+'/slurm.log/y_all.txt'; 

x_holdin_df.to_csv(txt_x_holdin_df,sep='\t',header=True,index_col=True);
x_holdout_df.to_csv(txt_x_holdout_df,sep='\t',header=True,index_col=True);
x_all.to_csv(txt_x_all,sep='\t',header=True,index_col=True);
y_holdin_df.to_csv(txt_y_holdin_df,sep='\t',header=True);
y_holdout_df.to_csv(txt_y_holdout_df,sep='\t',header=True);
y_all.to_csv(txt_y_all,sep='\t',header=True);

##########################################################
# distribute and run jobs	   
##########################################################

fid = open(filepath+'/slurm.log/'+simname+'.slurm','w');
fid.write('#!/bin/sh\n\n');
fid.write('#SBATCH --time-min=240\n');
fid.write('#SBATCH --mem=4096MB\n');
fid.write('#SBATCH --nice=500\n');
fid.write('#SBATCH --array=0-'+str(numperm-1)+'%180\n\n');
fid.write('source /home/lad44/davidlab/users/fsm/cholera/virtual_python_cholera/bin/activate\n\n');
fid.write('out_path='          +filepath+'/slurm.log/itr.$SLURM_ARRAY_TASK_ID.out\n');
fid.write('err_path='          +filepath+'/slurm.log/itr.$SLURM_ARRAY_TASK_ID.err\n\n');
fid.write('y_holdin_df='       +txt_y_holdin_df        +' \n');#+filepath+'/slurm.log/y.in.$SLURM_ARRAY_TASK_ID.txt'  +' \n');
fid.write('y_holdout_df='      +txt_y_holdout_df       +' \n');#+filepath+'/slurm.log/y.out.$SLURM_ARRAY_TASK_ID.txt' +' \n');
fid.write('y_all_df='          +txt_y_all              +' \n');#+filepath+'/slurm.log/y.all.$SLURM_ARRAY_TASK_ID.txt' +' \n');
fid.write('x_holdin_df='       +txt_x_holdin_df        +' \n');
fid.write('x_holdout_df='      +txt_x_holdout_df       +' \n');
fid.write('x_all_df='          +txt_x_all              +' \n');
fid.write('x_static_df='       +txt_clinical_df        +' \n');
fid.write('otu_taxa_map='      +txt_otu_taxa_map       +' \n');
fid.write('filepath='          +filepath               +' \n');
fid.write('simname='           +simname                +' \n');
fid.write('params='            +params                 +' \n');
fid.write('pickle_model='      +str(pickle_model)      +' \n');
fid.write('shuffle='           +str(shuffle)           +' \n');
fid.write('numperm='           +'$SLURM_ARRAY_TASK_ID' +' \n');
fid.write('myRandSeed='        +str(seedint)+'\n\n');
main_cmd = 'srun -o $out_path -e $err_path python '
main_cmd+=  pypath+'/class.single.versatile.two.stage.rfe.py ';
main_cmd+= '$y_holdin_df $y_holdout_df $y_all_df ';
main_cmd+= '$x_holdin_df $x_holdout_df $x_all_df $x_static_df $otu_taxa_map ';
main_cmd+= '$filepath $simname $params $pickle_model $shuffle $numperm $myRandSeed\n\n';
fid.write(main_cmd);
fid.write('echo $SLURM_ARRAY_JOB_ID > '+filepath+'/'+simname+'.job');
fid.close()

os.system('sbatch '+filepath+'/slurm.log/'+simname+'.slurm');
os.system('echo Job Array Submitted\n')

##########################################################
# end of script		   
##########################################################
