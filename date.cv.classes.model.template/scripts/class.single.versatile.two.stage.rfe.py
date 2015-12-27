#!/usr/bin/env python

# __author__ = Firas Said Midani
# __e-mail__ = fsm3@duke.edu
# ___date___ = 2015.12.07
# __version_ = 1.0

##########################################################
# DESCRIPTION  				   
##########################################################

# Given the following:
#  x_holdin_df
#  x_holdout_df
#  CVS
#  CVCLFS
#  num_features
#  coarse_steps     
#  fine_steps       
#  internal_cv      
#  include_otus
#  include_static
#  pickle_model
#
# Performs (1) cross-validated evaluation of model performance, (2) model training on 
#  a held-in subset of the data, (3) model evaluation on a held-out subset, and (4)
#  model training/evaluation on the whole data set
#  
# The script that tracks the AUROC for the cross-validation, model training, and model
#  validation is perfromed seperately seperately. 

##########################################################
# INPUT    				   
##########################################################

# txt_y_holdin_df       = sys.argv[1]; path to labels of hold in subset of data **
# txt_y_holdout_df      = sys.argv[2]; path to labels of hold out subset of data **
# txt_y_all             = sys.argv[3]; path to labels of the whole data set **
# txt_x_holdin_df       = sys.argv[4]; path to hold in subset of dynamic features matirx
# txt_x_holdout_df      = sys.argv[5]; path to hold out subset of dynamic featuers matrix
# txt_x_all             = sys.argv[6]; path to whole data set of dynamic features matrix
# txt_clinical_df       = sys.argv[7]; path to whole data set of static features (here clinical features) **
# filepath              = sys.argv[8]; filepath 
# simname               = sys.argv[9]; name of model 
# params                = sys.argv[10]; parameter file 
# pickle_model          = int(sys.argv[11]); 1="pickle select model output", 0="don't pickle anything"
# shuffle               = int(sys.argv[12]); 1="shuffle labels of training data", 0="don't shuffle anything"
# numperm		= int(sys.argv[13]); model iteration number
# myRandSeed            = int(sys.argv[14]); integer for seeding numpy random generator

# KEY
# ** these text file must include header of feature names and index_col of subject/sample IDs

##########################################################
# CREATES/MODIFIES/REMOVES   				   
##########################################################

#CREATES
#  * TXT (LOG) of python ouput for this main job
#  * PKL of classifier output 
#  *   scores_all: distance to hyperplane for each test sample
#  *   probas_all:                 P(y=1) for each test sample
#  *   predict_all:       predicted label for each test sample
#  *   tlabels_all:            true label for each test sample
#
#
#MODIFIES
# * TXT file logging AROC scores for classifier iterations related to this one
# * TXT file logging accuracy scores for classifier iterations related to this one
# * TXT file logging matthew's correlation coefficient scores  for classifier iterations related to this one
#
#
#REMOVES
# 
# 

##########################################################
# initialization 				   
##########################################################
					
import sys, imp, os

#####################################
# in-house tools
#####################################

pypath = os.path.dirname(os.path.realpath(sys.argv[0]));
foo = imp.load_source('classification_library',pypath+'/class.library.py');
from classification_library import *

pypath = os.path.dirname(os.path.relpath(sys.argv[0]));
foo    = imp.load_source('pruning_library',pypath+'/class.pruning.library.py');
from pruning_library import *

##########################################################
# seed random generator				   
##########################################################

#seed = random.randint(0, sys.maxint)
#myRand = random.Random(seed);

##########################################################
# input handling 				   
##########################################################

txt_y_holdin_df       = sys.argv[1];       print 'txt_y_holdin_df\t',       txt_y_holdin_df
txt_y_holdout_df      = sys.argv[2];       print 'txt_y_holdout_df\t',      txt_y_holdout_df
txt_y_all             = sys.argv[3];       print 'txt_y_all\t',             txt_y_all
txt_x_holdin_df       = sys.argv[4];       print 'txt_x_holdin_df\t',       txt_x_holdin_df
txt_x_holdout_df      = sys.argv[5];       print 'txt_x_holdout_df\t',      txt_x_holdout_df
txt_x_all             = sys.argv[6];       print 'txt_x_all\t',             txt_x_all
txt_clinical_df       = sys.argv[7];       print 'txt_clinical_df\t',       txt_clinical_df
filepath              = sys.argv[8];       print 'filepath\t',              filepath
simname               = sys.argv[9];      print 'simname\t',               simname
params                = sys.argv[10];      print 'parameter_file\t',        params 
pickle_model          = int(sys.argv[11]); print 'pickle_model\t',          pickle_model
shuffle               = int(sys.argv[12]); print 'shuffle\t',               shuffle
numperm               = int(sys.argv[13]); print 'numperm\t',		    numperm
myRandSeed            = int(sys.argv[14]); print 'myRandsed\t',             myRandSeed

foo = imp.load_source('model_parameters',params)
from model_parameters import *

# GENERATE RANDOM SEED FROM RANDOM INTEGER
random.seed(myRandSeed)

# INFER ITERATION NUMBER FROM INPUT NAME FORMAT
# numperm = txt_y_holdin_df.split('/')[-1].split('.')[2]

##########################################################
# classifier input
##########################################################

# INPUT FEATURE MATRIX AND ARRAY OF LABELS
x_holdin_df       = pd.read_csv(txt_x_holdin_df,       sep='\t',header=0,index_col=0);
x_holdout_df      = pd.read_csv(txt_x_holdout_df,      sep='\t',header=0,index_col=0);
x_all             = pd.read_csv(txt_x_all,             sep='\t',header=0,index_col=0);
y_holdin_df       = pd.read_csv(txt_y_holdin_df,       sep='\t',header=0,index_col=0);
y_holdout_df      = pd.read_csv(txt_y_holdout_df,      sep='\t',header=0,index_col=0);
y_all             = pd.read_csv(txt_y_all,             sep='\t',header=0,index_col=0);
clinical_df       = pd.read_csv(txt_clinical_df,       sep='\t',header=0,index_col=0);

#######################################################################
##Cross-validated model training to approximate generalized performance and feature importances
#######################################################################

if      CVS[0:3]=="LOO":
            cross_validation = LeaveOneOut(np.ravel(y_all).shape[0]);
elif    CVS[0:3]=="SSS":
            num_iterations   = int(CVS.split('.')[1]);
            test_pct         = float(CVS.split('.')[-1])/100;
            cross_validation = StratifiedShuffleSplit(np.ravel(y_all),n_iter=num_iterations,test_size=test_pct);
elif    CVS[0:3]=="SKF":
            num_folds        = int(CVS.split('.')[1]);
            cross_validation = StratifiedKFold(np.ravel(y_all),n_folds=num_folds); 
elif    CVS=="holdout_validation":
            cross_validation = [(np.array(range(0,x_holdin_df.shape[0])), np.array(range(x_holdin_df.shape[0],x_all.shape[0])))]; 
#endif

if   SCL[0:6]=='Normal':
	scaler = StandardScaler();	
elif SCL[0:6]=='MinMax':
	scaler = MinMaxScaler();
#endif

if   TSF=="SQRT":
	transformer = np.sqrt;
elif TSF=="LOG":
	transformer = lambda x: np.log10(x+1e-6);


print 'holdin\t',   x_holdin_df.shape,  y_holdin_df.shape
print 'holdout\t',  x_holdout_df.shape, y_holdout_df.shape
print 'all\t',      x_all.shape,        y_all.shape
print 'clinical\t', clinical_df.shape

print 'cross_validaiton\t',cross_validation
print 'num_features\t',num_features_1
print 'coarse_steps\t',coarse_steps_1
print 'CVCLFS\t',CVCLFS
print '(transform with transformer)\t',(transform,transformer)
print '(scale with scaler)\t',(scale,scaler)
print 'shuffle\t',shuffle

args_out = SVM_RFE_soft_two_stage(arg_ext_cv = cross_validation, \
				           x = x_all,\
					   y = y_all,\
		             static_features = clinical_df,\
			            coarse_1 = num_features_1,\
			       coarse_step_1 = coarse_steps_1,\
				         clf = CVCLFS,\
			    frequency_cutoff = frequency_cutoff,\
			           transform = transform,\
  				 transformer = transformer,\
				       scale = scale,\
    	 			      scaler = scaler,\
				include_otus = include_otus,\
  			      include_static = include_static,\
				     shuffle = shuffle);


cv_df_auc,cv_df_acc,cv_df_mcc,cv_df_features = [arg for arg in args_out]; 

if shuffle==0: 
	
	#######################################################################################
	##FINAL MODEL DESCRIPTION: Run classifier on all samples and record selected features
	#######################################################################################

	#initialize 
	df_features = pd.DataFrame(columns=["whole"]);
	
	################################################################################
	#Filter data based on frequency of presence of each feature across model samples
	#################################################################################
	bfd = pd.DataFrame(binarize(x_all),index=x_all.index,columns=x_all.keys())
	dense_features = bfd.keys()[np.where(bfd.apply(np.sum)>=np.ceil(frequency_cutoff*x_all.shape[0]))[0]]
	
	print 'thresholding'
	print x_all.shape,'-->',	
	x_use  = x_all.loc[:,dense_features];
	print x_use.shape

	#######################################################################################
	# Remove non informative redundant clades 
	#######################################################################################

	print 'pruning'
	print x_use.shape,'-->',
	x_use = dropNonInformativeClades(x_use);
	print x_use.shape

	#######################################################################################
	# Transform feature values
	#######################################################################################
	
	if transform==1:
		print 'transforming with ',transformer 
		x_use = x_use.apply(transformer)
		print np.mean(x_use.apply(np.sum,1))

	#######################################################################################
	# Scale feature arrays
	#######################################################################################
	
	if scale==1:
		print 'scaling with ',scaler
		x_use_scale = scaler.fit(x_use);
		x_use = pd.DataFrame(x_use_scale.transform(x_use), \
				     index=x_use.index, columns=x_use.keys());
	#endif

	x_use.to_csv(filepath+'/slurm.log/x_all_final_use.txt',sep='\t',header=True,index_col=True);
	
	if include_otus==1:
	
		# Narrow down full list of features
		rfe1             = RFE(estimator=CVCLFS,n_features_to_select=num_features_1,step=coarse_steps_1);
		coarse_features  = x_use.keys()[rfe1.fit(x_use,y_all).support_];
		x_use            = x_use.loc[:,coarse_features];

		# finely prune features one by one
		for num_feats in range(x_use.shape[1])[::-1][:-1]:
			print 'feature #'+str(num_feats)+'\t',			

			#single feature elimination
			SFE = RFE(CVCLFS,n_features_to_select=num_feats,step=1);
			SFE = SFE.fit(x_use,y_all);

			#track which feature is removed	
			features_kept               = x_use.keys()[SFE.support_];
			feature_removed             = x_use.keys()[~SFE.support_].values[0];
			df_features.loc[num_feats,'whole'] = feature_removed;
			print 'removed --> ',feature_removed				

			#transform featuer matrices
			x_use = x_use.loc[:,features_kept];
		#endfor
	#endif

# SAVE AUROC,ACCURACY,and MCC 
cv_df_auc.to_csv(filepath+'/cv_auc.txt',sep='\t',header=True,index_col=True);
cv_df_acc.to_csv(filepath+'/cv_acc.txt',sep='\t',header=True,index_col=True);
cv_df_mcc.to_csv(filepath+'/cv_mcc.txt',sep='\t',header=True,index_col=True);

# SAVE FEATURE LISTS/RANKING
cv_df_features.to_csv(filepath+'/cv_features.txt',sep='\t',header=True,index_col=True);
df_features.to_csv(filepath+'/features.txt',sep='\t',header=True,index_col=True);

##########################################################
# end of script		   
##########################################################
