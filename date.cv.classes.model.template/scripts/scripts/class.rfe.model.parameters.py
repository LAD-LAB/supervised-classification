import \
    pandas as pd, \
    matplotlib.pyplot as plt, \
    numpy as np,\
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
from sklearn.metrics            import auc, roc_curve

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
foo = imp.load_source('classification_library',pypath+'/class.library.py')
from classification_library import *

#####################################
# adding functionality to classifiers  
#####################################

class GridSearchWithCoef(GridSearchCV):
    #http://stackoverflow.com/questions/29538292/doing-hyperparameter-estimation-for-the-estimator-in-each-fold-of-recursive-feat
    @property
    def coef_(self):
        return self.best_estimator_.coef_
    def support_(self):
        return self.best_estimator_.support_
        
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


frequency_cutoff = 0;

num_features_1   = 350;
num_features_2   = 76;
coarse_steps_1   = 100;
coarse_steps_2   = 1;
fine_steps       = 1;
internal_cv      = 10; #folds
normalize        = 1;

#CVS = 'LOO';
#CVS = 'SSS.300.10';
#CVS = 'SKF.10';
CVS = 'holdout_validation';

CVCLFS   = LogisticRegression('l2',C=100)
#CVCLFS   = SVC(kernel='linear',probability=True,shrinking=True,cache_size=2000,C=100,random_state=24289074);

CLFS   = CVCLFS;

