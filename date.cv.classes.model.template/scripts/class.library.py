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
    pickle, \
    random, \
    time, \
    sys, \
    imp, \
    os
    
#####################################
# classification auxiliary tools
#####################################

from sklearn.cross_validation   import StratifiedKFold, LeaveOneOut, StratifiedShuffleSplit 
from sklearn.feature_selection  import RFECV, RFE, SelectPercentile, SelectKBest
from sklearn.grid_search        import GridSearchCV
from sklearn.preprocessing      import StandardScaler, MinMaxScaler, binarize
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

####################################
#
#####################################
# additional in-house tools
#####################################

pypath = os.path.dirname(os.path.relpath(sys.argv[0]));
foo    = imp.load_source('pruning_library',pypath+'/class.pruning.library.py');
from pruning_library import *

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

def SVM_RFE_soft_two_stage(**kwargs):
    
    cv,clf                                = [kwargs.get(varb) for varb in ['arg_ext_cv','clf']];
    x,y,static_features                   = [kwargs.get(varb) for varb in ['x','y','static_features']];
    coarse_1,coarse_step_1                = [int(kwargs.get(varb)) for varb in ['coarse_1','coarse_step_1']];
    frequency_cutoff                      = [float(kwargs.get(varb)) for varb in ['frequency_cutoff']][0]; 
    include_otus,include_static           = [int(kwargs.get(varb)) for varb in ['include_otus','include_static']];  
    shuffle,scale,transform               = [int(kwargs.get(varb)) for varb in ['shuffle','scale','transform']];
    scaler,transformer                    = [kwargs.get(varb) for varb in ['scaler','transformer']];

    print 'num_features\t',coarse_1
    print 'coarse steps\t',coarse_step_1
    print 'frequency_cutoff\t',frequency_cutoff 
    print '(include_otus,include_static)\t(',include_otus,',',include_static,')'
    print '(scale with scaler)\t',scale,scaler
    print '(transform with transformer)\t',transform,transformer
    print 'shuffle\t',shuffle
	
    # initialize
    _tests_ix,_trues,_scores,_probas,_predicts,_support,_ranking,_auroc_p,_auroc_s,_acc,_mcc  = [[] for aa in range(11)];
    df_auc,df_acc,df_mcc,df_features = [pd.DataFrame(columns=range(1,len(cv)+1)) for aa in range(4)];
	
    # initialize recursive feature elimination objects
    rfe1     = RFE(estimator=clf,n_features_to_select=coarse_1,step=coarse_step_1)  
    pnl_coef = {};    

    cnt = 0;
    for train, test in cv:
        cnt+=1; print "%04.f" %cnt,
	
	# split data ino training and testing folds
        x_train,x_test,y_train,y_test = split_only(x,y,train,test);

	# shuffle labels if requested
	if shuffle==1:
		np.random.shuffle(y_train.values);

	#################################################################################
	#Filter data based on frequency of presence of each feature across model samples
	#################################################################################
	bfd = pd.DataFrame(binarize(x_train),index=x_train.index,columns=x_train.keys())
	dense_features = bfd.keys()[np.where(bfd.apply(np.sum)>=np.ceil(frequency_cutoff*x_train.shape[0]))[0]]
	
	print 'thresholding'
	print '(train,test)\t',x_train.shape,x_test.shape,'-->',
	x_train = x_train.loc[:,dense_features]
	x_test  = x_test.loc[:,dense_features]
	print x_train.shape,x_test.shape

	#################################################################################
	#Remove non-informative redundnat clades
	#################################################################################

	print 'pruning'
	print '(train,test)\t',x_train.shape,x_test.shape,'-->',
	x_train = dropNonInformativeClades(x_train)	
	x_test  = x_test.loc[:,x_train.keys()];	
	print x_train.shape,x_test.shape

	#################################################################################
	#Transform feature values
	#################################################################################

	if transform==1:
		print 'transforming with ',transformer
		x_train = x_train.apply(transformer);
		x_test  = x_test.apply(transformer);

	#################################################################################
	#Scale feature arrays
	#################################################################################
	
	if scale==1:
                print 'scaling with ',scaler
		x_train_scale = scaler.fit(x_train);
		x_train       = pd.DataFrame(x_train_scale.transform(x_train), \
					     index=x_train.index, columns=x_train.keys());
		x_test        = pd.DataFrame(x_train_scale.transform(x_test),  \
					     index=x_test.index, columns=x_test.keys());
	      	
	#################################################################################
	#Apply Recrusive Feature Elimination wrapped aroud a user-defined classifier
	#################################################################################
	
	# use only features that are transformed with RFE
	if include_otus==1: 
	
	 	# Narrow down  full list of features to 100                                     
	        coarse_features = x_train.keys()[rfe1.fit(x_train,y_train).support_] 
       		x_train         = x_train.loc[:,coarse_features];
		
		df_coef = pd.DataFrame(index=x_train.keys(), columns=range(1,x_train.shape[1])); 

		# finely prune features one by one
		for num_feats in range(x_train.shape[1])[::-1][:-1]:
		        print 'feature #'+str(num_feats)+'\t', 
	
			#single feature elimination
			SFE = RFE(clf,n_features_to_select=num_feats,step=1);
			SFE = SFE.fit(x_train,y_train);

			# track which feature is removed
			features_kept           = x_train.keys()[SFE.support_];
			feature_removed         = x_train.keys()[~SFE.support_].values[0];
			df_features.loc[num_feats,cnt] = feature_removed;
			print 'removed --> ',feature_removed,'\t',

			# transform feature matrices
			x_train  = x_train.loc[:,features_kept];
			x_test   = x_test.loc[:,features_kept];      		

			# join non-filtered (static) featuers
			if include_static==1:
				x_train = x_train.join(static_features);
				x_test  = x_test.join(static_features);
			#endif

			# fit and test classifier with remaining featuers (store AUC)
			clf_fit  = clf.fit(x_train,y_train);
			clf_eval = clf_fit.decision_function(x_test);
			clf_pdct = clf_fit.predict(x_test);

			# compute AUC, accuracy, and MCC
			clf_auc  = roc_auc_score(y_test,clf_eval);
			clf_acc  = accuracy_score(y_test,clf_pdct);
			clf_mcc  = matthews_corrcoef(y_test,clf_pdct);

			# record model performance
			df_auc.loc[num_feats,cnt] = clf_auc;
			df_acc.loc[num_feats,cnt] = clf_acc;
			df_mcc.loc[num_feats,cnt] = clf_mcc;
			
			print '==> (AUC,ACC,MCC) = (',
			print ('%0.4f,' % clf_auc),
			print ('%0.4f,' % clf_acc),
			print ('%0.4f,' % clf_mcc),
			print ')'

			# record model coefficients
			clf_coef = clf_fit.coef_[0];
			clf_vars = x_train.keys();
			for varb,coef in zip(clf_vars,clf_coef):
				df_coef.loc[varb,num_feats]=coef;
			#endfor
		#endfor
	
	# use only static unselected features
	elif  (include_otus==0) and (include_static==1):

		x_train   = static_features.loc[x_train.index,:];
		x_test    = static_features.loc[x_test.index,:];
	
		df_coef   = pd.DataFrame(index=x_train.keys(), columns=['clinical']);

		# fit and test classifier with remaining featuers (store AUC)
		clf_fit  = clf.fit(x_train,y_train);
		clf_eval = clf_fit.decision_function(x_test);
		clf_pdct = clf_fit.predict(x_test);

		# compute AUC, accuracy, and MCC
		clf_auc  = roc_auc_score(y_test,clf_eval);
		clf_acc  = accuracy_score(y_test,clf_pdct);
		clf_mcc  = matthews_corrcoef(y_test,clf_pdct);

		# record model performance
		df_auc.loc[num_feats,cnt] = clf_auc;
		df_acc.loc[num_feats,cnt] = clf_acc;
		df_mcc.loc[num_feats,cnt] = clf_mcc;
			
		print '==> (AUC,ACC,MCC) = (',
		print ('%0.4f,' % clf_auc),
		print ('%0.4f,' % clf_acc),
		print ('%0.4f,' % clf_mcc),
		print ')'

		# record model coefficients
		clf_coef = clf_fit.coef_[0];
		clf_vars = x_train.keys();
		for varb,coef in zip(clf_vars,clf_coef):
			df_coef.loc[varb,'clinical']=coef;
		#endfor
	#endif	
		
			
	pnl_coef[cnt] = df_coef;

    return df_auc,df_acc,df_mcc,df_features,pnl_coef
