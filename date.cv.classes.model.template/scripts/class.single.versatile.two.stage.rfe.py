#!/usr/bin/env python

# __author__ = Firas Said Midani
# __e-mail__ = fsm3@duke.edu
# ___date___ = 2015.12.07
# __version_ = 1

##########################################################
# DESCRIPTION  				   
##########################################################

# Given the following:
#  x_holdin_df
#  x_holdout_df
#  x_holdin_norm_df
#  x_holdout_norm_df
#  CVS
#  CVCLFS
#  CLFS
#  num_features
#  coarse_steps     
#  fine_steps       
#  internal_cv      
#  normalize        
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

# params                = sys.argv[1]; parameter file 
# txt_y_holdin_df       = sys.argv[2]; path to labels of hold in subset of data **
# txt_y_holdout_df      = sys.argv[3]; path to labels of hold out subset of data **
# txt_y_all             = sys.argv[4]; path to labels of the whole data set **
# simname               = sys.argv[5]; name of model 
# txt_x_holdin_df       = sys.argv[6]; path to hold in subset of dynamic features matirx
# txt_x_holdout_df      = sys.argv[7]; path to hold out subset of dynamic featuers matrix
# txt_x_all             = sys.argv[8]; path to whole data set of dynamic features matrix
# txt_x_holdin_norm_df  = sys.argv[9]; same as txt_x_holdin_df but standard normalized features **
# txt_x_holdout_norm_df = sys.argv[10]; same as txt_x_holdout_df but standard normalized features **
# txt_clinical_df       = sys.argv[11]; path to whole data set of static features (here clinical features) **
# filepath              = sys.argv[12]; filepath 
# include_otus          = int(sys.argv[13]); 1="include dynamic (bacterial) features in model", 0="don't"
# include_static        = int(sys.argv[14]); 1="include static (clinical) features in model", 0="don't"
# pickle_model         = int(sys.argv[15]); 1="pickle certain output of model", 0="don't"
# myRandSeed            = int(sys.argv[16]); integer for seeding numpy random generator

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
foo = imp.load_source('classification_library',pypath+'/class.library.py')
from classification_library import *

######################################
# modification of scikit-learn objects
######################################
class GridSearchWithCoef(GridSearchCV):
    #http://stackoverflow.com/questions/29538292/doing-hyperparameter-estimation-for-the-estimator-in-each-fold-of-recursive-feat
    @property
    def coef_(self):
        return self.best_estimator_.coef_
    def support_(self):
        return self.best_estimator_.support_

##########################################################
# seed random generator				   
##########################################################

#seed = random.randint(0, sys.maxint)
#myRand = random.Random(seed);

##########################################################
# input handling 				   
##########################################################

params                = sys.argv[1]; print 'txt_X\t',params 
txt_y_holdin_df       = sys.argv[2]; print 'txt_y_holdin_df\t',txt_y_holdin_df
txt_y_holdout_df      = sys.argv[3]; print 'txt_y_holdout_df\t',txt_y_holdout_df
txt_y_all             = sys.argv[4]; print 'txt_y_all\t',txt_y_all
simname               = sys.argv[5]; print 'simname\t',simname
txt_x_holdin_df       = sys.argv[6]; print 'txt_x_holdin_df\t',txt_x_holdin_df
txt_x_holdout_df      = sys.argv[7]; print 'txt_x_holdout_df\t',txt_x_holdout_df
txt_x_all             = sys.argv[8]; print 'txt_x_all\t',txt_x_all
txt_x_holdin_norm_df  = sys.argv[9]; print 'txt_x_holdin_norm_df\t',txt_x_holdin_norm_df
txt_x_holdout_norm_df = sys.argv[10]; print 'txt_x_holdout_norm_df\t',txt_x_holdout_norm_df
txt_clinical_df       = sys.argv[11]; print 'txt_clinical_df\t',txt_clinical_df
filepath              = sys.argv[12]; print 'filepath\t',filepath
include_otus          = int(sys.argv[13]); print 'include_otus\t',include_otus
include_static        = int(sys.argv[14]); print 'include_static\t',include_static
pickle_model         = int(sys.argv[15]); print 'pickle_model\t',pickle_model
myRandSeed            = int(sys.argv[16]); #integer for seeding numpy random generator

foo = imp.load_source('model_parameters',params)
from model_parameters import *

# GENERATE RANDOM SEED FROM RANDOM INTEGER
random.seed(myRandSeed)

# INFER ITERATION NUMBER FROM INPUT NAME FORMAT
numperm = txt_y_holdin_df.split('/')[-1].split('.')[2]

##########################################################
# classifier input
##########################################################

# INPUT FEATURE MATRIX AND ARRAY OF LABELS
x_holdin_df       = pd.read_csv(txt_x_holdin_df,       sep='\t',header=0,index_col=0);
x_holdout_df      = pd.read_csv(txt_x_holdout_df,      sep='\t',header=0,index_col=0);
x_all             = pd.read_csv(txt_x_all,             sep='\t',header=0,index_col=0);
x_holdin_norm_df  = pd.read_csv(txt_x_holdin_norm_df,  sep='\t',header=0,index_col=0);
x_holdout_norm_df = pd.read_csv(txt_x_holdout_norm_df, sep='\t',header=0,index_col=0);
y_holdin_df       = pd.read_csv(txt_y_holdin_df,       sep='\t',header=0,index_col=0);
y_holdout_df      = pd.read_csv(txt_y_holdout_df,      sep='\t',header=0,index_col=0);
y_all             = pd.read_csv(txt_y_all,             sep='\t',header=0,index_col=0);
clinical_df       = pd.read_csv(txt_clinical_df,       sep='\t',header=0,index_col=0);

#x_all = pd.concat([x_holdin_df,x_holdout_df])
#y_all = pd.concat([y_holdin_df,y_holdout_df])


#######################################################################
#Filter data based on frequency of presence of each feature across model samples
#######################################################################
#bfd = pd.DataFrame(binarize(x_all),index=x_all.index,columns=x_all.keys())
#dense_features = bfd.keys()[np.where(bfd.apply(np.sum)>=np.ceil(frequency_cutoff*x_all.shape))[0]]

x_all_norm_df, x_all_norm_df = standard_normalize_training_data_and_transform_validation_data(\
                                              x_all,x_all)

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

print 'holdin\t',   x_holdin_df.shape,  y_holdin_df.shape
print 'holdout\t',  x_holdout_df.shape, y_holdout_df.shape
print 'all\t',      x_all.shape,        y_all.shape
print 'clinical\t', clinical_df.shape

print 'cross_validaiton\t',cross_validation
print 'x_all_norm_df,y_all\t',x_all_norm_df.shape,y_all.shape
print 'num_features\t',num_features_1,num_features_2
print 'coarse_steps\t',coarse_steps_1,coarse_steps_2
print 'fine_steps\t', fine_steps
print 'CVCLFS,CLFS\t',CVCLFS,CLFS
print 'normalize\t',normalize

cv_trues,cv_probas,cv_predicts,cv_scores,_auroc_p,_auroc_s,_acc,_mcc = SVM_RFE_soft_two_stage(cross_validation,x_all,y_all,\
											       num_features_1,num_features_2,\
											       coarse_steps_1,coarse_steps_2,fine_steps,internal_cv,\
											       CVCLFS,frequency_cutoff,normalize,clinical_df,\
											       include_otus,include_static);
                                                             
#_trues,_probas,_scores,_select = [arg for arg in cv_args_out[0:4]];
fpr_p,tpr_p,thresh_p = roc_curve(np.ravel(cv_trues),cv_probas);
fpr_s,tpr_s,thresh_s = roc_curve(np.ravel(cv_trues),cv_scores);

cv_auroc_p     = auc(fpr_p,tpr_p);
cv_auroc_s     = auc(fpr_s,tpr_s);

cv_acc         = accuracy_score(np.ravel(cv_trues),cv_predicts);
cv_mcc         = matthews_corrcoef(np.ravel(cv_trues),cv_predicts);
 
#######################################################################
###STAGE 1 MODEL TRAINING: Classifier Trained on held-in training  set with reduced set of features 
#######################################################################
training_probas,training_scores,training_trues = [[] for aa in range(3)]

if normalize==1:
    x_holdin_scaled_df  = x_holdin_norm_df;
    x_holdout_scaled_df = x_holdout_norm_df;
else:
    x_holdin_scaled_df  = x_holdin_df;
    x_holdout_scaled_df = x_holdout_df;

if include_otus==1:

	rfe1 = RFE(estimator=CVCLFS,n_features_to_select=num_features_1,step=coarse_steps_1)
	rfe2 = RFE(estimator=CVCLFS,n_features_to_select=num_features_2,step=coarse_steps_2)

	coarse_selector_1 = rfe1.fit(x_holdin_scaled_df,y_holdin_df);
	coarse_selector_2 = rfe2.fit(x_holdin_scaled_df.iloc[:,coarse_selector_1.support_],y_holdin_df)

	top_features       = np.where(coarse_selector_1.support_)[0][np.where(coarse_selector_2.support_)[0]];
	top_features_names = x_holdin_scaled_df.keys()[top_features];

	x_holdin_top  = x_holdin_scaled_df.iloc[:,top_features];
	x_holdout_top = x_holdout_scaled_df.iloc[:,top_features].fillna(0);
  
if    (include_otus==1) and (include_static==1): 
	x_holdin_top  = x_holdin_top.join(clinical_df)
	x_holdout_top = x_holdout_top.join(clinical_df).fillna(0);

elif  (include_otus==0) and (include_static==1):
	x_holdin_top  = clinical_df.loc[x_holdin_df.index];
	x_holdout_top = clinical_df.loc[x_holdout_df.index].fillna(0);

opt_clf = CLFS.fit(x_holdin_top,y_holdin_df);

training_probas+= list(opt_clf.predict_proba(x_holdin_top)[:,1])
training_scores+= list(opt_clf.decision_function(x_holdin_top))
training_trues += list(y_holdin_df.values); 
fpr_p,tpr_p,thresh_p  = roc_curve(np.ravel(training_trues),training_probas);
fpr_s,tpr_s,thresh_s  =	roc_curve(np.ravel(training_trues),training_scores);
training_auroc_p      = auc(fpr_p,tpr_p)
training_auroc_s      = auc(fpr_s,tpr_s)
training_acc          = accuracy_score(y_holdin_df.values, opt_clf.predict(x_holdin_top));
training_mcc          = matthews_corrcoef(y_holdin_df.values, opt_clf.predict(x_holdin_top));

#######################################################################
###STAGE 2 MODEL VALIDATION: Classifier Tested on held-out validation set iwht reduced set of features
#######################################################################
validation_probas,validation_scores,validation_trues = [[] for aa in range(3)]

# Evaluate Model Performance on held out data
validation_probas+= list(opt_clf.predict_proba(x_holdout_top)[:,1])
validation_scores+= list(opt_clf.decision_function(x_holdout_top))
validation_trues += list(y_holdout_df.values); 
fpr_p,tpr_p,thresh_p  = roc_curve(np.ravel(validation_trues),validation_probas);
fpr_s,tpr_s,thresh_s  = roc_curve(np.ravel(validation_trues),validation_scores);
validation_auroc_p    = auc(fpr_p,tpr_p)
validation_auroc_s    = auc(fpr_s,tpr_s)
validation_acc        = accuracy_score(y_holdout_df.values, opt_clf.predict(x_holdout_top));
validation_mcc        = matthews_corrcoef(y_holdout_df.values, opt_clf.predict(x_holdout_top));

#######################################################################################
##STAGE 3 MODEL DESCRIPTION: Run classifier on all samples and record selected features
#######################################################################################

if normalize==1:
	x_use = x_all_norm_df;
else: 
	x_use = x_all;
#endif

#######################################################################
#Filter data based on frequency of presence of each feature across model samples
#######################################################################
bfd = pd.DataFrame(binarize(x_all),index=x_all.index,columns=x_all.keys())
dense_features = bfd.keys()[np.where(bfd.apply(np.sum)>=np.ceil(frequency_cutoff*x_all.shape[0]))[0]]

x_use  = x_use.loc[:,dense_features]
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

######################################################################
# summarize and record classifier results			   
##########################################################

# PICKLE CLASSIFIER RESULTS
# topickle = ['scores_all','probas_all','predict_all','tlabels_all','auroc','features_tally'];			
# PIK = filepath+'/slurm.log/'+str(numperm)+'.output.pickle';
# with open(PIK,"wb") as f:
# 	pickle.dump(topickle,f)
# 	for value in topickle:
# 		pickle.dump([scores_all, probas_all, predict_all,tlabels_all,auroc,features_tally],f)   

# SAVE AUROC 
fid = open(filepath+'/auroc.txt','a');
fid.write(str(numperm)+'\t'+str(cv_auroc_p)         +'\t'+str(cv_auroc_s)         + \
                       '\t'+str(training_auroc_p)   +'\t'+str(training_auroc_s)   + \
		       '\t'+str(validation_auroc_p) +'\t'+str(validation_auroc_s) + \
 		       '\t'+str(np.mean(_auroc_p))  +'\t'+str(np.mean(_auroc_s))  +'\n')
fid.close()

# SAVE MCC
fid = open(filepath+'/acc.txt','a');
fid.write(str(numperm)+'\t'+str(cv_acc)+'\t'+str(training_acc)+'\t'+str(validation_acc)+'\t'+str(np.mean(_acc))+'\n')
fid.close()

# LOG SUCCESSFUL COMPLETION OF ITERATION
fid = open(filepath+'/mcc.txt','a');
fid.write(str(numperm)+'\t'+str(cv_mcc)+'\t'+str(training_mcc)+'\t'+str(validation_mcc)+'\t'+str(np.mean(_mcc))+'\n')
fid.close()

##########################################################
# end of script		   
##########################################################
