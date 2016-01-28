#!/usr/bin/env python

# ___author___ = Firas Said Midani
# ___e-mail___ = firas.midani@duke.edu
# ___date_____ = 2050;12.07
# ___version__ = 1.0

#####################################
# INITIALIZATION
#####################################
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

num_features_1   = 501;
coarse_steps_1   = 10;

transform=1;
transform_static=1;

scale=1;
scale_static=1;

include_otus=1;
include_static=0;
include_static_with_prob=0;

# CHOOSE VALIDATION SCHEME

# 'LOO','holdout_validation','SSS.100.10'
CVS = 'SKF.10';

# CHOOSE SCALER (only applies if normalize==1)
# 'Normal','MinMax';
SCL='Normal';
SCLSTATIC='Normal';
scale_static_varbs = ['vbxbase','ageyrs','bloodo'];

# CHOOSE TRANSFORM
# 'SQRT','LOG';
TSF='LOG';
TSFSTATIC='LOG';
transform_static_varbs = ['vbxbase',];

## CHOOSE CLASSIFIER
CLSS='svm.l1';
CLSSTATIC='log.l2';
