#!/usr/bin/env python

# __author__ = Firas Said Midani
# __e-mail__ = firas.midani@duke.edu
# __update__ = 2015.12.07
# __version_ = 1.0

#############################################################################
# DESCRIPTION
#############################################################################
# This script summarizes the results of a series of classifiers that vary by
#  simply a number (here NF). It summarizes the mean metrics for the 
#  empirical model and its respective permuted iterations. The latter allows
#  for the computation of a p-value (or Type I error) for the empirical 
#  model. 

########################################################
# DESCRIPTION OF SUMMARY ESTIMATES
#########################################################

# Mean
# 	Each cross-validation has K-folds. This is the mean of the corresponding K ROC curves. If cross validation is resampled (say M times), this is further averaged across all resamples. 
#       =\frac{1}{M*K} * \sum_{i=1}^{M*K} AuROC_{i}
#
# ModelPooled
# 	Each cross-validation has K-folds. The probability estimtes (or decision scores) are pooled across the K folds and used to comute a gingle ROC curve for the cross vaidation. 
# 	If cross validation is resamples( say M times), this is further averaged across all resamples.
#       =roc_curve(true_labels_{i=1...M*K*(N/K)},probability_Estiamtes_{i=1...*M*K*(n/K)}; where N = # of samples in data
#
# Averaged_Estimates_Emp_Metric
# 	Each cross-validation has K-folds. The probability estimate (or decision score) are pooled across the K folds. If cross validation is resampels, the sample-specific 
#	probability estimates (or decision scores) are averaged across all resamples. This genreates averaged sample-specific estimates (or scores) which are used to generate an ROC curve. 
#       for each sample in n=1...N 
#       	probability_estimate_n = \frac{1}{M} \sum_{i=1}^{M} probability_estimate_{i}
#       =roc_curve(true_labels_{i=1...N},probability_estimate_n{i=1...N}))
#  
# Emp_Metric_LowerBound, Emp_Metric_Median, Emp_Metric_UpperBound
#       The ModelAveraged-based ROC curve is further bootstrapped and its lower bound, median, and upper bound are recorded. 
# 
# MeanNull
# 	Similar as Mean, but uses the Null Models where the labels in training subsets are shuffled. This value is averaged over all permutations of the Null Model 
# 	
# ModelPooledNull
#	Similar as Pooled_Estimates_Emp_metric, but uses the Null Models wehre the labels in training subsets are shuffled. This value is averaged over all permutations of the Null model.
#
# NumIterations
#	Number of iterations of the empirical model (typically 1)
# 	
# NumNullIterations
#       Number of iterations of the null models (typically at least 50)
#
# MeanPvalue
#       P-value of Mean 
#
# ModelPooledPvalue
#       P-value of ModelPooled
#

#############################################################################
# INITIALIZATION AND ARGUMENT HANDLINGN
#############################################################################

import \
pandas as pd, \
numpy  as np, \
subprocess, \
time, \
sys, \
imp, \
os

pypath = os.path.dirname(os.path.realpath(sys.argv[0]));
foo    = imp.load_source('classification_summary_library',pypath+'/class.summary.library.py')
from classification_summary_library import *

filepath  = sys.argv[1] 	  # parent directory for classifier runs 
metric    = sys.argv[2] 	  # either of {auroc, acc, mcc} which correspond to area under ROC curve, accuracy, and matthews's correlation coefficient scores
if len(sys.argv)>3:     	  # if you select auroc, you also need to define the underlying numbers to compute it. either of the following
	roc_frame = sys.argv[3]   # {probas, scores, mean_probas, mean_scores}
	bootstrap = int(sys.argv[4])
#endif

summary_pnl = pd.DataFrame(columns=['Mean',\
				    'MeanNull',\
				    'MeanPvalue',\
				    'ModelPooled',\
			            'ModelPooledNull',\
		   	            'ModelPooledPvalue',\
				    'ModelAveraged',\
				    'ModelAveraged_LowerBound',\
				    'ModelAveraged_Median',\
				    'ModelAveraged_UpperBound',\
			            'NumIterations',\
			            'NumNullIterations']).astype(float);

if metric=="auroc":
	if   roc_frame == "probas":
		idx_p1 = 0;
		idx_p2 = 2;
	elif roc_frame == "scores":
		idx_p1 = 1;
		idx_p2 = 3;
	#endif
#endif

for nf in range(1,501):

	##########################################################
	# COMPUTE SUMMARY ESTIMATES OF EMPIRICAL MODELS
	##########################################################
	empirical = filepath + '/results/class.two.stage.rfe.'+str(nf)+'.empirical/'+metric+'.txt';
	if os.path.isfile(empirical):
		if os.stat(empirical).st_size != 0:
			nf_summ = pd.read_csv(empirical,sep='\t',header=None,index_col=0);
			summary_pnl.loc[nf,'Mean']           =              np.mean(nf_summ.iloc[0,idx_p2]);
			summary_pnl.loc[nf,'ModelPooled']    =              np.mean(nf_summ.iloc[0,idx_p1]);
			summary_pnl.loc[nf,'NumIterations']  = str(int(len(np.where(nf_summ.iloc[:,idx_p1])[0]))); 
			Mean        = nf_summ.iloc[0,idx_p2];
			ModelPooled = nf_summ.iloc[0,idx_p1];
		#endif
	#endif

	##########################################################
	# COMPUTE BOOTSTRAPPED ESTIMATES OF CLASSIFIER PERFORMANCE
	##########################################################
	
	if bootstrap == 1:	
		empirical = filepath + '/results/class.two.stage.rfe.'+str(nf)+'.empirical/slurm.log/itr.0.pickle';
		if os.path.isfile(empirical):
			
			# initialization
			n_bootstraps = 3000;
			BS_auc_, BS_acc_, BS_mcc_                = [[] for ii in range(3)];
			trues_df, scores_df, probas_df, preds_df = [pd.DataFrame() for ii in range(4)]; 
			
			pickle_path = filepath+'/results/class.two.stage.rfe.'+str(nf)+'.empirical/slurm.log/';
	
			# Find out how many resamples of cross validation are saved
			cmd = ["ls "+pickle_path+"itr*pickle | awk -F '/' '{print $NF}' | awk -F '.' '{print $2}' "];
			model_iterations = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE).communicate()[0].rstrip().split('\n')
			
			# for each resample, grab pooled true labels, their label scores, probability estimates, and predicted labels
			for itr in model_iterations:
				trues_dict, scores_dict, probas_dict, preds_dict = grabClassModelFitFromPickle(pickle_path+'itr.'+str(itr)+'.pickle');
	
				trues_df[itr],scores_df[itr],probas_df[itr],preds_df[itr] = [pd.Series(ii,name=itr) for ii in [trues_dict,scores_dict,probas_dict,preds_dict]];
			
			# average the probability estimates and decisions cores across all resamples of cross vlidation
			list_dict                     = [trues_df,scores_df,probas_df,preds_df];
			list_labels                   = ['trues','mean_scores','mean_probas','mean_predictions'];
			y_true,y_fit_s,y_fit_p,y_pred = [pd.Series(x.apply(np.mean,1),name=y) for x,y in zip(list_dict,list_labels)];
			
			pd.DataFrame([y_true,y_fit_s,y_fit_p,y_pred]).to_csv(pickle_path+'model_predictions_df.txt',sep='\t',header=True,index_col=True);

			# bootstrap the averaged decision scores to bootstrap the ROC curve and its corresponding area	
			y_score = y_fit_s.values;
			y_pred  = y_pred.values;
			y_true  = y_true.values;
	
			y_true_df, y_pred_df, y_fit_df = bootstrapClassifierFit(y_true,y_pred,y_score,n_bootstraps,True)	 
			
			for i in range(n_bootstraps):
				BootstrappedClassifierPerformance = ClassifierPerformance(y_true_df.iloc[:,i], y_pred_df.iloc[:,i], y_fit_df.iloc[:,i]);
				BS_auc_.append(BootstrappedClassifierPerformance.Metrics().auc)
		
			confidence_lower, confidence_median, confidence_upper = bootstrappedConfidenceIntervals(BS_auc_,0.05);

			# record bootstraped confidence lower, median, and upper bounds
			summary_pnl.loc[nf,'ModelAveraged_LowerBound'] = confidence_lower;
			summary_pnl.loc[nf,'ModelAveraged_UpperBound'] = confidence_upper; 
			summary_pnl.loc[nf,'ModelAveraged_Median']     = confidence_median;
	
	##########################################################
	# COMPUTE SUMMARY ESTIMATES OF NULL MODELS
	##########################################################
	permutation = filepath + '/results/class.two.stage.rfe.'+str(nf)+'.permutations/'+metric+'.txt';
	if os.path.isfile(permutation):
		if os.stat(permutation).st_size != 0:
			nf_summ = pd.read_csv(permutation,sep='\t',header=None,index_col=0);
			summary_pnl.loc[nf,'MeanNull']           =              np.mean(nf_summ.iloc[:,idx_p2]);
			summary_pnl.loc[nf,'ModelPooledNull']    =              np.mean(nf_summ.iloc[:,idx_p1]);
			summary_pnl.loc[nf,'NumNullIterations']  = str(int(len(np.where(nf_summ.iloc[:,idx_p1])[0])));
			summary_pnl.loc[nf,'MeanPvalue']         =   float(len(np.where(nf_summ.iloc[:,idx_p2]>Mean)[0]));
			summary_pnl.loc[nf,'MeanPvalue']        /=        (len(np.where(nf_summ.iloc[:,idx_p2])[0])+1);
			summary_pnl.loc[nf,'ModelPooledPvalue']  =   float(len(np.where(nf_summ.iloc[:,idx_p1]>ModelPooled)[0]));
			summary_pnl.loc[nf,'ModelPooledPvalue'] /=        (len(np.where(nf_summ.iloc[:,idx_p1])[0])+1);
		#endif
	#endif

summary_pnl.to_csv(filepath+'/summary/class.two.stage.rfe.summary.'+metric+'.txt',sep='\t',header=True,index_col=True,float_format='%0.4f')

