import sys, imp, os

#####################################
# in-house tools
#####################################

pypath = os.path.dirname(os.path.realpath(sys.argv[0]));
foo = imp.load_source('classification_library',pypath+'/class.library.py')
from classification_library import *

#######################################################################
## Model parameters
#######################################################################
feature_options  = {'description':['phylum', 'class', 'order', 'family', 'genus', 'species', 'OTU']}
#feature_options  = {'description':['phylum', 'class', 'order', 'family', 'genus', 'species']}
#feature_options  = {'description':['species']}

holdin_flag1     = {'color':[1,2],'day':[2],'Batch':[1]}
holdin_flag2     = {'color':[3,4],'day':[2],'Batch':[1]}

holdout_flag1     = {'color':[1,2],'day':[2],'Batch':[2]}
holdout_flag2     = {'color':[3,4],'day':[2],'Batch':[2]}

outcomes_dict    = {1:1,2:1,3:0,4:0}
outcomes_dict    = {1:1,2:1,3:0,4:0}


frequency_cutoff = 0.10;

num_features_1   = 350;
num_features_2   = 76;
coarse_steps_1   = 100;
coarse_steps_2   = 1;
fine_steps       = 1;
internal_cv      = 10; #folds
normalize        = 1;
include_otus     = 1;
include_static   = 0;
pickle_model     = 0;

#CVS = 'LOO';
CVS = 'SSS.30.10';
#CVS = 'SKF.10';
#CVS = 'holdout_validation';

CVCLFS   = LogisticRegression('l2',C=100)
#CVCLFS   = SVC(kernel='linear',probability=True,shrinking=True,cache_size=2000,C=100,random_state=24289074);

CLFS   = CVCLFS;

