#!/usr/bin/env python

# __author__ = Firas Said Midani
# __e-mail__ = fsm3@duke.edu
# __date__   = 2015.09.06
# __update__ = 2015.09.06

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
#  - internal cross-validation number of folds
#  - cross-validation scheme
#  - classification method and parameterization
# 
# Do the following:
#  - Perform either multiple iterations of a classifier, with true labels or shuffled labels. 
#  - Output a histogram of AuROC with n equal to the number of iterations/permutations chosen. 
# 
# Designed to run a HPC (e.g. SLURM). It will create a job array with each corresponding
#  to a single classification (with selected cross-validation scheme, classifier, and 
#  parameter grid search. See svm.seperate.py for more details. 

#########################################################
# INPUT 
#########################################################

# params      = sys.argv[1]; parameter file
# filepath    = sys.argv[2]; path to store reulsts
# simname     = sys.argv[3]; simulation name (used for naming files)
# shuffle     = sys.argv[4]; 1="shuffle labels", 0="don't shuffle labels"
# numperm     = int(sys.argv[5]); "number of permutations/iterations

##########################################################
# CREATES/MODIFIES/REMOVES
##########################################################

# CREATES
#  * DIR SLURM.LOG 
#  * TXT summary of this main job
#  * TXT (OUT) of shell output for this main job
#  * TXT (ERR) of shell errors for this main job
#  * TXT (SLURM) shell commands for submitting job array
#  * TXT tracking successful job completion ~~
#  * TXT summary of jobs running on SLURM
#  * TXT list of jobs held by SLURM
   #  * TXT of feature matrix
#  * PKL of random seed for this main job 
#  * PDF of histogram of AUROC for all iterations
#
#  times number of iterations (jobs submitted by this script)
   #  * TXT of labels for each iteration 
#  * TXT (OUT) of shell output for each iteration 
#  * TXT (ERR) of shell errors for each iteration 
#  * TXT (OUT) file on directory where this main job is submitted (not sure why duplicate)
#  * TXT (ERR) file on directory where this main job is submitted (not sure why duplicate)
#
#
# REMOVES
# * TXT tracking successful job completion ~~
#
#
# ~~ these are files created and removed by this script

##########################################################
# dependencies
##########################################################

#  cholera_classifier_lib: e.g. 2015.02.28.smic.lib.py
#  releaseHeldJobs.py
#  SLURM: designed for usage on a HPC.
 
##########################################################
# initialization 				   
##########################################################
					
import warnings
warnings.filterwarnings('ignore')

import \
    pandas as pd, \
    matplotlib.pyplot as plt, \
    numpy as np,\
    random, \
    time, \
    imp, \
    sys, \
    os

#####################################
# classification auxiliary tools
#####################################

from sklearn.cross_validation   import StratifiedKFold, LeaveOneOut, StratifiedShuffleSplit 
from sklearn.feature_selection  import RFECV, RFE, SelectKBest
from sklearn.grid_search        import GridSearchCV
from sklearn.preprocessing      import StandardScaler, binarize
from sklearn.metrics            import auc, roc_curve, roc_auc_score

#####################################
# classification algorithms
#####################################

from sklearn.linear_model       import LogisticRegression
from sklearn.svm                import SVC
from sklearn.ensemble           import RandomForestClassifier

#####################################
# prototyping tools
#####################################

from sklearn.datasets           import make_classification

#####################################
# statistical tests
#####################################

from scipy.stats                import kruskal, mannwhitneyu

#####################################
# saving plots on pdf
#####################################

from   matplotlib.ticker               import MultipleLocator, FormatStrFormatter
from   matplotlib.backends.backend_pdf import PdfPages

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
txt_mapping        = sys.argv[4]; print 'txt_mapping\t',txt_mapping 
txt_clinical_df    = sys.argv[5]; print 'txt_clinical\t',txt_clinical_df
filepath           = sys.argv[6]; print 'filepath\t',filepath
simname            = sys.argv[7]; print 'simname\t',simname
shuffle            = int(sys.argv[8]); print 'shuffle:\t',shuffle
numperm            = int(sys.argv[9]); print 'numperm\t',numperm
include_otus       = int(sys.argv[10]); print 'include_otus\t',include_otus
include_static     = int(sys.argv[11]); print 'include_static\t',include_static
pickle_model       = int(sys.argv[12]); print 'pickle_model\t',pickle_model

foo = imp.load_source('model_parameters',params)
from model_parameters import *

##########################################################
# classifier input and parameters			   
##########################################################

#######################################################################
##Read feature matrix and feature meta-data
#######################################################################
features_df    = pd.read_csv(txt_features_df,sep='\t',header=0,index_col=0)
taxa_level_map = pd.read_csv(txt_taxa_level_map,sep='\t',header=0,index_col=0)

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
##Filter data based on frequency of presence of each feature across model samples
#######################################################################
bfd = pd.DataFrame(binarize(x_holdin_df),index=x_holdin_df.index,columns=x_holdin_df.keys())
dense_features = bfd.keys()[np.where(bfd.apply(np.sum)>=np.ceil(frequency_cutoff*x_holdin_df.shape[0]))[0]]

x_holdin_df    = x_holdin_df.loc[:,dense_features]
x_holdout_df   = x_holdout_df.loc[:,dense_features]

x_holdin_norm_df,x_holdout_norm_df = standard_normalize_training_data_and_transform_validation_data(x_holdin_df.copy(),x_holdout_df.copy());

##############################################################
# initialization for tracking of jobs' completion and results 				   
##############################################################

# INITIALIZE LOG OF SUCCESSFUL JOB MAIN OUTPUT AUROC
for filetype in ['/auroc.txt','/acc.txt','/mcc.txt']:
	fid = open(filepath+filetype,'w+');
	fid.close()

##########################################################
# distribute jobs input		   
##########################################################

# INITIALIZE JOB ARRAY REPOSITORY
if not os.path.isdir(filepath+'/slurm.log'):
    os.system('mkdir '+filepath+'/slurm.log');
    
# SAVE FEATURE MATRIX (should be same for all jobs in array)
txt_x_holdin_df       = filepath+'/slurm.log/x_holdin_df.txt';
txt_x_holdout_df      = filepath+'/slurm.log/x_holdout_df.txt';
txt_x_all             = filepath+'/slurm.log/x_all.txt';
txt_x_holdin_norm_df  = filepath+'/slurm.log/x_holdin_norm_df.txt';
txt_x_holdout_norm_df = filepath+'/slurm.log/x_holdout_norm_df.txt';
txt_y_holdin_df       = filepath+'/slurm.log/y_holdin_df.txt';
txt_y_holdout_df      = filepath+'/slurm.log/y_holdout_df.txt';
txt_y_all             = filepath+'/slurm.log/y_all.txt'; 

x_holdin_df.to_csv(txt_x_holdin_df,sep='\t',header=True,index_col=True);
x_holdout_df.to_csv(txt_x_holdout_df,sep='\t',header=True,index_col=True);
x_all.to_csv(txt_x_all,sep='\t',header=True,index_col=True);
x_holdin_norm_df.to_csv(txt_x_holdin_norm_df,sep='\t',header=True,index_col=True);
x_holdout_norm_df.to_csv(txt_x_holdout_norm_df,sep='\t',header=True,index_col=True);
y_holdin_df.to_csv(txt_y_holdin_df,sep='\t',header=True);
y_holdout_df.to_csv(txt_y_holdout_df,sep='\t',header=True);
y_all.to_csv(txt_y_all,sep='\t',header=True);

# SAVE LABELS ARRAY 
for perm in range(numperm):	
	pi_in  = y_holdin_df;
	pi_out = y_holdout_df;
	pi_all = y_all;
	if shuffle==1:	
		np.random.shuffle(pi_in);
	pi_in.to_csv(filepath+'/slurm.log/y.in.'+str(perm)+'.txt',sep='\t',header=True);
	pi_out.to_csv(filepath+'/slurm.log/y.out.'+str(perm)+'.txt',sep='\t',header=True);
	pi_all.to_csv(filepath+'/slurm.log/y.all.'+str(perm)+'.txt',sep='\t',header=True);

##########################################################
# distribute and run jobs	   
##########################################################

fid = open(filepath+'/slurm.log/'+simname+'.slurm','w');
fid.write('#!/bin/sh\n\n');
fid.write('#SBATCH --time-min=240\n');
fid.write('#SBATCH --mem=4096MB\n');
fid.write('#SBATCH --nice=300\n');
#fid.write('#SBATCH --job-name==\n');
#fid.write('#SBATCH --dependency=singleton\n');
fid.write('#SBATCH --array=0-'+str(numperm-1)+'%180\n\n');
fid.write('out_path='+filepath+'/slurm.log/itr.$SLURM_ARRAY_TASK_ID.out\n');
fid.write('err_path='+filepath+'/slurm.log/itr.$SLURM_ARRAY_TASK_ID.err\n\n');
main_cmd = 'srun -o $out_path -e $err_path python '+pypath+'/class.single.versatile.rfe.py ';
main_cmd+= params+' ';
main_cmd+= filepath+'/slurm.log/y.in.$SLURM_ARRAY_TASK_ID.txt ';
main_cmd+= filepath+'/slurm.log/y.out.$SLURM_ARRAY_TASK_ID.txt ';
main_cmd+= filepath+'/slurm.log/y.all.$SLURM_ARRAY_TASK_ID.txt ';
main_cmd+= simname+' ';
main_cmd+= txt_x_holdin_df+' ';
main_cmd+= txt_x_holdout_df+' ';
main_cmd+= txt_x_all+' ';
main_cmd+= txt_x_holdin_norm_df+' ';
main_cmd+= txt_x_holdout_norm_df+' ';
main_cmd+= txt_clinical_df+' ';
main_cmd+= filepath+' ';
main_cmd+= str(include_otus)+' ';
main_cmd+= str(include_static)+' ';
main_cmd+= str(pickle_model)+' ';
main_cmd+= str(seedint)+'\n\n';
fid.write(main_cmd);
fid.write('echo $SLURM_ARRAY_JOB_ID > '+filepath+'/'+simname+'.job');
fid.close()

os.system('sbatch '+filepath+'/slurm.log/'+simname+'.slurm');
os.system('echo Job Array Submitted\n')

##########################################################
# end of script		   
##########################################################
