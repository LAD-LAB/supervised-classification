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
    
    cv,clf,clf_static                     = [kwargs.get(varb) for varb in ['arg_ext_cv','clf','clf_static']];
    x,y,static_features                   = [kwargs.get(varb) for varb in ['x','y','static_features']];
    coarse_1,coarse_step_1                = [int(kwargs.get(varb)) for varb in ['coarse_1','coarse_step_1']];
    frequency_cutoff                      = [float(kwargs.get(varb)) for varb in ['frequency_cutoff']][0]; 
    include_otus,include_static           = [int(kwargs.get(varb)) for varb in ['include_otus','include_static']];  
    shuffle,scale,transform,pickle_model  = [int(kwargs.get(varb)) for varb in ['shuffle','scale','transform','pickle_model']];
    scaler,transformer                    = [kwargs.get(varb) for varb in ['scaler','transformer']];
    include_static_with_prob              = [kwargs.get(varb) for varb in ['include_static_with_prob']][0];	
    filepath    			  = [kwargs.get(varb) for varb in ['filepath']][0];
    numperm     			  = [str(kwargs.get(varb)) for varb in ['numperm']][0];

 
    print 'num_features\t',coarse_1
    print 'coarse steps\t',coarse_step_1
    print 'frequency_cutoff\t',frequency_cutoff 
    print '(include_otus,include_static)\t(',include_otus,',',include_static,')'
    print '(scale with scaler)\t',scale,scaler
    print '(transform with transformer)\t',transform,transformer
    print 'shuffle\t',shuffle
	
    # initialize
    _tests_ix,_trues,_scores,_probas,_predicts,_support,_ranking,_auroc_p,_auroc_s,_acc,_mcc  = [[] for aa in range(11)];
    
    cnt = int(numperm)*len(cv);
    #df_auc,df_acc,df_mcc = [pd.DataFrame(columns=range(cnt,cnt+len(cv)+1)) for aa in range(3)];
    #df_auc,df_acc,df_mcc = [pd.DataFrame(columns=range(1,len(cv)+1)) for aa in range(3)];
	
    # initialize recursive feature elimination objects
    rfe1     = RFE(estimator=clf,n_features_to_select=coarse_1,step=coarse_step_1)  
    
    for train, test in cv:
        cnt+=1; print "%04.f" %cnt,
	
	# split data ino training and testing folds
        x_train,x_test,y_train,y_test = split_only(x,y,train,test);

	# shuffle labels if requested
	if shuffle==1:
		np.random.shuffle(y_train.values);

	# use only features that are transformed with RFE
	if include_otus==1: 

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
	
	 	# Narrow down  full list of features to 100                                     
	        coarse_features = x_train.keys()[rfe1.fit(x_train,y_train).support_] 
       		x_train         = x_train.loc[:,coarse_features];

		# initializes frames for recording data
		df_auc,df_acc,df_mcc = [pd.DataFrame(index=range(1,x_train.shape[1]),columns=[cnt]) for aa in range(3)];		
		df_features          = pd.DataFrame(index=x_train.keys(), columns=[cnt]);	
		df_prob              = pd.DataFrame(index=x_test.index,   columns=range(1,x_train.shape[1]));
		
		if   include_static==0:
			df_coef     = pd.DataFrame(index=x_train.keys(), columns=range(1,x_train.shape[1])); 
		elif include_static==1:
			df_coef     = pd.DataFrame(index=list(x_train.keys())+list(static_features.keys()), columns=range(1,x_train.shape[1]));

		# finely prune features one by one
		for num_feats in range(x_train.shape[1])[::-1][:-1]:

			x_train_keys = list(set(x_train.keys()).difference(static_features.keys()))			
			x_train      = x_train.loc[:,x_train_keys];

			#single feature elimination
			SFE = RFE(clf,n_features_to_select=num_feats,step=1);
			SFE = SFE.fit(x_train,y_train);

			# track which feature is removed
			features_kept           = x_train.keys()[SFE.support_];
			feature_removed         = x_train.keys()[~SFE.support_].values[0];
			df_features.loc[feature_removed,cnt] = num_feats+1;
			
			# transform feature matrices
			x_train  = x_train.loc[:,features_kept];
			x_test   = x_test.loc[:,features_kept];      		

			# join non-filtered (static) featuers
			if include_static==1:
				x_train = x_train.join(static_features,how='left');
				x_test  = x_test.join(static_features,how='left');
			#endif

			# fit and test classifier with remaining featuers (store AUC)
			clf_fit  = clf.fit(x_train,y_train);
			clf_eval = clf_fit.decision_function(x_test);
			clf_pdct = clf_fit.predict(x_test);
			clf_coef = clf_fit.coef_[0];

			#if include_static_with_prob==1:
	
			#	x_all    = x_train.append(x_test);
			#	clf_eval = pd.DataFrame(clf_fit.decision_function(x_all),index=x_all.index,columns=['bacterial_risk']);
			#	x_all    = x_all.join(static_features,how='left');
			#	
			#	x_train_tmp  = x_all.loc[y_train.index,:];
			#	x_test_tmp   = x_all.loc[y_test.index,:];

			#	clf_fit  = clf_static.fit(x_train_tmp,y_train_tmp);
			#	clf_eval = clf_fit.decision_function(x_test_tmp);
			#	clf_pdct = clf_fit.predict(x_test_tmp);
			#	clf_coef = clf_fit.coef_[0];

			# compute AUC, accuracy, and MCC
			df_auc.loc[num_feats,cnt]  = roc_auc_score(y_test,clf_eval);
			df_acc.loc[num_feats,cnt]  = accuracy_score(y_test,clf_pdct);
			df_mcc.loc[num_feats,cnt]  = matthews_corrcoef(y_test,clf_pdct);

			# record model coefficients
			df_coef.loc[x_train.keys(),num_feats] = clf_coef;
			
			# record model estimates of P(y=1) for subjects
			df_prob.loc[x_test.index,num_feats] = clf_eval;
		#endfor
	
	# use only static unselected features
	elif  (include_otus==0) and (include_static==1):

		x_train   = static_features.loc[x_train.index,:];
		x_test    = static_features.loc[x_test.index,:];
		
		df_auc,df_acc,df_mcc = [pd.DataFrame(index=['clinical'],columns=[cnt]) for aa in range(3)];
 		df_coef              = pd.DataFrame(index=x_train.keys(), columns=['clinical']);
		df_prob              = pd.DataFrame(index=x_test.index,  columns=['clinical']);

		# fit and test classifier with remaining featuers (store AUC)
		clf_fit  = clf_static.fit(x_train,y_train);
		clf_eval = clf_fit.decision_function(x_test);
		clf_pdct = clf_fit.predict(x_test);
		clf_coef = clf_fit.coef_[0];

		# compute AUC, accuracy, and MCC
		df_auc.loc['clinical',cnt] = roc_auc_score(y_test,clf_eval);
		df_acc.loc['clinical',cnt] = accuracy_score(y_test,clf_pdct);
		df_mcc.loc['clinical',cnt] = matthews_corrcoef(y_test,clf_pdct);

		# record model coefficients
		df_coef.loc[x_train.keys(),'clinical'] = clf_coef;
			
		# record model estimates of P(y=!) for subjects
		df_prob.loc[x_test.index,'clinical'] = clf_eval;
	
	#endif

	# SAVE AUROC,ACCURACY,and MCC 
	df_auc.to_csv(filepath+'/slurm.log/itr.'+numperm+'.cvfold.'+str(cnt)+'.auc.txt',sep='\t',header=True,index_col=True);
	df_acc.to_csv(filepath+'/slurm.log/itr.'+numperm+'.cvfold.'+str(cnt)+'.acc.txt',sep='\t',header=True,index_col=True);
	df_mcc.to_csv(filepath+'/slurm.log/itr.'+numperm+'.cvfold.'+str(cnt)+'.mcc.txt',sep='\t',header=True,index_col=True);
	
	df_coef.to_csv(filepath+'/slurm.log/itr.'+numperm+'.cvfold.'+str(cnt)+'.coef.txt',sep='\t',header=True,index_col=True);
	df_prob.to_csv(filepath+'/slurm.log/itr.'+numperm+'.cvfold.'+str(cnt)+'.prob.txt',sep='\t',header=True,index_col=True);
	
	if include_otus:
		df_features.to_csv(filepath+'/slurm.log/itr.'+numperm+'.cvfold.'+str(cnt)+'.features.txt',sep='\t',header=True,index_col=True);
