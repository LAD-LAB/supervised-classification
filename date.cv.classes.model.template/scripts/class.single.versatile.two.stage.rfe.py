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
#  CLFS
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
# txt_clinical_df       = sys.argv[9]; path to whole data set of static features (here clinical features) **
# txt_otu_taxa_map      = sys.argv[10]; path to OTU-->Taxonomy map
# filepath              = sys.argv[11]; filepath 
# simname               = sys.argv[12]; name of model 
# params                = sys.argv[13]; parameter file 
# num_features_2        = int(sys.argv[14]); target number of features in predictive model
# pickle_model          = int(sys.argv[15]); 1="pickle select model output", 0="don't pickle anything"
# shuffle               = int(sys.argv[16]); 1="shuffle labels of training data", 0="don't shuffle anything"
# numperm		= int(sys.argv[17]); model iteration number
# myRandSeed            = int(sys.argv[18]); integer for seeding numpy random generator

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
txt_otu_taxa_map      = sys.argv[8];       print 'txt_otu_taxa_map\t',      txt_otu_taxa_map
filepath              = sys.argv[9];       print 'filepath\t',              filepath
simname               = sys.argv[10];      print 'simname\t',               simname
params                = sys.argv[11];      print 'parameter_file\t',        params 
num_features_2        = int(sys.argv[12]); print 'num_features_2\t',        num_features_2
pickle_model          = int(sys.argv[13]); print 'pickle_model\t',          pickle_model
shuffle               = int(sys.argv[14]); print 'shuffle\t',               shuffle
numperm               = int(sys.argv[15]); print 'numperm\t',		    numperm
myRandSeed            = int(sys.argv[16]); print 'myRandsed\t',             myRandSeed

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
otu_taxa_map      = pd.read_csv(txt_otu_taxa_map,      sep='\t',header=0,index_col=0);

#######################################################################
##Cross-validated model training to approximate generalized performance and feature importances
#######################################################################

if      CVS[0:3]=="LOO":
            cross_validation = LeaveOneOut(y_all.shape[0]);
elif    CVS[0:3]=="SSS":
            num_iterations   = int(CVS.split('.')[1]);
            test_pct         = float(CVS.split('.')[-1])/100;
            cross_validation = StratifiedShuffleSplit(y_all,n_iter=num_iterations,test_size=test_pct);
elif    CVS[0:3]=="SKF":
            num_folds        = int(CVS.split('.')[1]);
            cross_validation = StratifiedKFold(y_all,n_folds=num_folds); 
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
	transformer = np.log10;


print 'holdin\t',   x_holdin_df.shape,  y_holdin_df.shape
print 'holdout\t',  x_holdout_df.shape, y_holdout_df.shape
print 'all\t',      x_all.shape,        y_all.shape
print 'clinical\t', clinical_df.shape

print 'cross_validaiton\t',cross_validation
print 'num_features\t',num_features_1,num_features_2
print 'coarse_steps\t',coarse_steps_1,coarse_steps_2
print 'fine_steps\t', fine_steps
print 'CVCLFS,CLFS\t',CVCLFS,CLFS
print '(transform with transformer)\t',(transform,transformer)
print '(scale with scaler)\t',(scale,scaler)
print 'shuffle\t',shuffle

args_out = SVM_RFE_soft_two_stage(arg_ext_cv = cross_validation, \
				           x = x_all,\
					   y = y_all,\
		             static_features = clinical_df,\
			            coarse_1 = num_features_1,\
				    coarse_2 = num_features_2,\
			       coarse_step_1 = coarse_steps_1,\
			       coarse_step_2 = coarse_steps_2,\
                                   fine_step = fine_steps,\
                                  arg_int_cv = internal_cv,\
				         clf = CVCLFS,\
			    frequency_cutoff = frequency_cutoff,\
			           transform = transform,\
  				 transformer = transformer,\
				       scale = scale,\
    	 			      scaler = scaler,\
			        otu_taxa_map = otu_taxa_map,\
				include_otus = include_otus,\
  			      include_static = include_static,\
				     shuffle = shuffle);


cv_tests_ix,cv_trues,cv_predicts,cv_probas,cv_scores,_auroc_p,_auroc_s,_acc,_mcc = [arg for arg in args_out]; 

fpr_p,tpr_p,thresh_p = roc_curve(np.ravel(cv_trues),cv_probas);
fpr_s,tpr_s,thresh_s = roc_curve(np.ravel(cv_trues),cv_scores);

cv_auroc_p     = auc(fpr_p,tpr_p);
cv_auroc_s     = auc(fpr_s,tpr_s);

cv_acc         = accuracy_score(np.ravel(cv_trues),cv_predicts);
cv_mcc         = matthews_corrcoef(np.ravel(cv_trues),cv_predicts);

if shuffle==0: 
	
	#######################################################################################
	##FINAL MODEL DESCRIPTION: Run classifier on all samples and record selected features
	#######################################################################################
	
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
	x_use = dropNonInformativeClades(x_use,otu_taxa_map);
	print x_use.shape

	#######################################################################################
	# Transform feature values
	#######################################################################################
	
	if transform==1:
		x_use = x_use.apply(transformer)

	#######################################################################################
	# Scale feature arrays
	#######################################################################################
	
	if scale==1:
		x_use_scale = scaler.fit(x_use);
		x_use = pd.DataFrame(x_use_scale.transform(x_use), \
				     index=x_use.index, columns=x_use.keys());
	else: 
		x_use = x_all;
	#endif

	x_use.to_csv(filepath+'/slurm.log/x_all_final_use.txt',sep='\t',header=True,index_col=True);
	
	if include_otus==1:
		rfe1 = RFE(estimator=CVCLFS,n_features_to_select=num_features_1,step=coarse_steps_1)
		rfe2 = RFE(estimator=CVCLFS,n_features_to_select=num_features_2,step=coarse_steps_2)
	
		coarse_selector_1 = rfe1.fit(x_use,y_all);
		coarse_selector_2 = rfe2.fit(x_use.iloc[:,coarse_selector_1.support_],y_all)
	
		top_features       = np.where(coarse_selector_1.support_)[0][np.where(coarse_selector_2.support_)[0]];
		top_features_names = x_use.keys()[top_features];
		print 'top_features',top_features_names.values
	
		x_all_top = x_use.iloc[:,top_features];
		x_all_top.to_csv(filepath+'/slurm.log/x_all_top.txt',sep='\t',header=True,index_col=True);
	
	#endif

	######################################################################
	# summarize and record classifier results			   
	##########################################################
	
	# (cv_scores,   cv_probas)   pooled decision_function scores and predict_proba probabilities respectively
	# (cv_predicts, cv_trues)    pooled classifier predictions of labels and actual labels respectively
	# (_auroc_s,    _auroc_p)    Areas under the ROC curve for each classifier validation iterations (does not pool scores/probas across iterations)
	
	# PICKLE CLASSIFIER RESULTS
	topickle = ['cv_scores',   'cv_probas',                \
	            'cv_predicts', 'cv_trues', 'cv_tests_ix',  \
	            '_auroc_s',    '_auroc_p'];
	
	PIK = filepath+'/slurm.log/itr.'+str(numperm)+'.pickle';
	with open(PIK,"wb") as f:
		pickle.dump(topickle,f)
		for value in topickle:
			pickle.dump([cv_scores,   cv_probas,              \
	     			     cv_predicts, cv_trues,  cv_tests_ix,  \
	   			     _auroc_s,    _auroc_p],              \
	      		           f)   

# SAVE AUROC 
fid = open(filepath+'/auroc.txt','a');
fid.write(str(numperm)+'\t'+str(cv_auroc_p)         +'\t'+str(cv_auroc_s)         + \
 		       '\t'+str(np.mean(_auroc_p))  +'\t'+str(np.mean(_auroc_s))  +'\n')
fid.close()

# SAVE MCC
fid = open(filepath+'/acc.txt','a');
fid.write(str(numperm)+'\t'+str(cv_acc)+'\t'+str(np.mean(_acc))+'\n')
fid.close()

# LOG SUCCESSFUL COMPLETION OF ITERATION
fid = open(filepath+'/mcc.txt','a');
fid.write(str(numperm)+'\t'+str(cv_mcc)+'\t'+str(np.mean(_mcc))+'\n')
fid.close()

##########################################################
# end of script		   
##########################################################
