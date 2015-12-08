#!/usr/bin/env python -W ignore::DeprecationWarning
#
# ___author___ : Firas Said Midani
# ___e-mail___ : firas.midani@duke.edu
# ___date_____ : 2015.12.07
# ___version__ : 1.0
#
# LIST OF FUNCTIONS
#
# douleprint
#	tunnels python output to a text file as well as shell
#
# microbeAbbreviate
# 	simplifies bacterial feature names
#
# split_only
#       splits a feature matrix and lables into training and testing subsets
#
# standard_normalize_training_data_and_transform_validation_data(df)
#	for each variable (column) in a training dataframe, identify normal scale (mu and sigma)
#   then, transform accordingly the training, validation, and full data
#
# subset_data
#	identifies samples that match a list of user-defined critiera. requires a mapping file.
#     
# summarize_with_roc
# 	summarize classifier performance with an ROC curve
#
# SVM_RFE_soft_two_stage
# 	performs cross-validated classification with linear SVM on optimized number of 
#   features based on recursive feature elimination

import warnings
warnings.filterwarnings('ignore')

import \
    pandas as pd, \
    numpy as np,\
    random, \
    time, \
    sys, \
    os
    
#####################################
# classification auxiliary tools
#####################################

from sklearn.cross_validation   import StratifiedKFold, LeaveOneOut, StratifiedShuffleSplit 
from sklearn.feature_selection  import RFECV, RFE, SelectPercentile, SelectKBest
from sklearn.grid_search        import GridSearchCV
from sklearn.preprocessing      import StandardScaler, binarize
from sklearn.metrics            import auc, roc_curve, roc_auc_score, accuracy_score, matthews_corrcoef

#####################################
# classification algorithms
#####################################

from sklearn.linear_model       import LogisticRegression
from sklearn.svm                import SVC
from sklearn.ensemble           import RandomForestClassifier

#####################################
# statistical tests
#####################################

from scipy.stats                import mannwhitneyu

#####################################

def doubleprint(logfid,msg):
	# DOUBLEPRINT allows you to print string "msg" to your screen console (or code cell) 
	# and simultaneously prints output to file with handle "logfid"
    logfid.write(msg+"\n")

#####################################

def microbeAbbreviate(fullTaxonomy):
    # given full taxonomy, give acronym
    #     lowest = fullTaxonomy.split("__")[-1];
    #     second_lowest = fullTaxonomy.split("__")[-2].split(";")[0];
    #     acronym = second_lowest+" "+lowest;
    #     return acronym
    
    precision_levels = {"k__":"kingdom","p__":"phylum","c__":"class","o__":"order",\
                        "f__":"family","g__":"genus","s__":"species"}
    
    split_by_underscore = fullTaxonomy.split("__")
    split_taxonomy=[];
    cnt=0;
    for s in split_by_underscore:
        if s.split(";")[0] =='':
            cnt=cnt+1;
        split_taxonomy.append(s.split(";")[0])
        
    reducedTaxonomy=[];
    for x in range(1,cnt+3):
        if split_taxonomy[-(x)]=="":
            reducedTaxonomy.append(split_by_underscore[-(1+x)].split(";")[1]+"__")
        else:
            reducedTaxonomy.append(split_taxonomy[-x])
    reducedTaxonomy = " ".join(reducedTaxonomy[::-1])
    
    precision = fullTaxonomy.split(';')[-1].strip()[0:3]
        
    if precision !='s__':   
        reducedTaxonomy = reducedTaxonomy+" ("+precision_levels[precision]+")"
        
    return reducedTaxonomy

#####################################

def split_only(x,y,train,test):    

    x_train = x.iloc[train,:]#.values;
    x_test  = x.iloc[test,:]#.values;
    
    y_train = y.iloc[train]#.values;
    y_test  = y.iloc[test]#.values;
    
    return x_train,x_test,y_train,y_test
 
#####################################

def standard_normalize_training_data_and_transform_validation_data(training_df,validation_df):
	
	normal_scale       = StandardScaler().fit(training_df);
	training_norm_df   = normal_scale.transform(training_df);
	training_norm_df   = pd.DataFrame(training_norm_df, \
									  index   = training_df.index, \
									  columns = training_df.keys());
	
	validation_norm_df = normal_scale.transform(validation_df);
	validation_norm_df = pd.DataFrame(validation_norm_df, \
	                                  index   = validation_df.index, \
	                                  columns = validation_df.keys());
	
	return training_norm_df, validation_norm_df
	

#####################################

def subset_data(list_of_criteria,mapping_df,features_df):
    # list_of_criteria is a dictionary of variables and their desired values
    # mapping_df must include the variables from teh list_of_criteria 
    # features_df is the feature design matrix (samples x features)
    
    all_matches = pd.DataFrame();
    
    for lc in list_of_criteria:
        # identify samples in the meeet desired criteria (lc)
        matching_subset = mapping_df.loc[:,lc.keys()]
        for (key,value) in lc.iteritems():
    		matching_subset.loc[:,key] = [row in value for row in matching_subset.loc[:,key]];
    		
    	matching_subset = mapping_df.loc[matching_subset.all(1)]

        # identify matching samples that have features in the features_df matrix
        matching_subset = matching_subset.loc[matching_subset.index.isin(features_df.index),:]
        
        all_matches = pd.concat([all_matches,matching_subset])
    
    return all_matches 
    
#####################################

def SVM_RFE_soft_two_stage(arg_ext_cv,x,y,coarse_1,coarse_2,coarse_step_1,coarse_step_2,fine_step,arg_int_cv,clf,frequency_cutoff,normalize,static_features,include_otus,include_static):


    cv = arg_ext_cv

    # initialize
    _trues,_scores,_probas,_predicts,_support,_ranking,_auroc_p,_auroc_s,_acc,_mcc  = [[] for aa in range(10)]

    rfe1    = RFE(estimator=clf,n_features_to_select=coarse_1,step=coarse_step_1)  
    rfe2    = RFE(estimator=clf,n_features_to_select=coarse_2,step=coarse_step_2)  

    cnt = 0;
    for train, test in cv:
        cnt+=1; print cnt,
        x_train,x_test,y_train,y_test = split_only(x,y,train,test)


	#######################################################################
	#Filter data based on frequency of presence of each feature across model samples
	#######################################################################
	bfd = pd.DataFrame(binarize(x_train),index=x_train.index,columns=x_train.keys())
	dense_features = bfd.keys()[np.where(bfd.apply(np.sum)>=np.ceil(frequency_cutoff*x_train.shape[0]))[0]]

	x_train = x_train.loc[:,dense_features]
	x_test  = x_test.loc[:,dense_features]


	if normalize==1:
        	x_train, x_test = standard_normalize_training_data_and_transform_validation_data(x_train,x_test);
      	
	if include_otus==1: 
	 	# Narrow down  full list of features to 100                                     
	        coarse_selector_1 = rfe1.fit(x_train,y_train)  
        	coarse_selector_2 = rfe2.fit(x_train.iloc[:,coarse_selector_1.support_],y_train)  
                
 		# Identify the top features that maximizes AUROC
        	top_features  = np.where(coarse_selector_1.support_)[0][np.where(coarse_selector_2.support_)[0]]
       
		x_train_top   = x_train.iloc[:,top_features]#.join(static_features);
		x_test_top    = x_test.iloc[:,top_features]#.join(static_features);

	if    (include_otus==1) and (include_static==1):
		x_train_top   = x_train_top.join(static_features);
		x_test_top    = x_test_top.join(static_features);
	elif  (include_otus==0) and (include_static==1):
		x_train_top   = static_features.loc[x_train.index,:];
		x_test_top    = static_features.loc[x_test.index,:];

        # Train model using only those features 
        opt_clf         = clf.fit(x_train_top,y_train);

        # Evaluate model performance with testing data
        probas   = opt_clf.predict_proba(x_test_top)[:,1]; 
	scores   = opt_clf.decision_function(x_test_top);
	predicts = opt_clf.predict(x_test_top);
	trues    = y_test.values;
	
	_probas   += list(probas);
 	_predicts += list(predicts);
	_scores   += list(scores);
	_trues    += list(trues);

        if len(test)>2:
		auroc_p, auroc_s   = (roc_auc_score(trues,probas),    roc_auc_score(trues,scores));
        	accuracy, matthews = (accuracy_score(trues,predicts), matthews_corrcoef(trues,predicts)); 
        	
		_auroc_p.append(auroc_p)
		_auroc_s.append(auroc_s)
		_acc.append(accuracy)
		_mcc.append(matthews)

		print "roc_p=%0.4f" % auroc_p
		print "roc_s=%0.4f" % auroc_s
		print "acc=%0.4f"   % accuracy
		print "mcc=%0.4f"   % matthews
		
    return _trues,_probas,_predicts,_scores,_auroc_p,_auroc_s,_acc,_mcc
