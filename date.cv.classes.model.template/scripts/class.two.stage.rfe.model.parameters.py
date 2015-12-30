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
transform=0;
scale=0;
include_otus=1;
include_static=0;
include_static_with_prob=0;

# CHOOSE VALIDATION SCHEME

#CVS = 'LOO';
#CVS = 'holdout_validation';
#CVS = 'SKF.10';
CVS = 'SSS.100.10';

# CHOOSE SCALER (only applies if normalize==1)
SCL=;
#SCL='MinMax';

# CHOOSE TRANSFORM
#TSF='LOG';
TSF=;

## CHOOSE CLASSIFIER
CLSS=;
CLSSTATIC=;
