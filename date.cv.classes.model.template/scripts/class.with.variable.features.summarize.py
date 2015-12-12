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

import \
pandas as pd, \
numpy  as np, \
time, \
sys, \
os

pypath = os.path.dirname(os.path.realpath(sys.argv[0]));
foo    = imp.load_source('classification_summary_library',pypath+'/class.summary.library.py')
from classification_summary_library import *

filepath  = sys.argv[1] # parent directory for classifier runs 
metric    = sys.argv[2] # either of {auroc, acc, mcc} which correspond to area under ROC curve, accuracy, and matthews's correlation coefficient scores
if len(sys.argv)>3:     # if you select auroc, you also need to define the underlying numbers to compute it. either of the following
	roc_frame = sys.argv[3]   # {probas, scores, mean_probas, mean_scores}
#endif

summary_pnl = pd.DataFrame(columns=['AvgEmpMetric','AvgNullMetric','NumEmpIterations','NumNullIterations','EmpMetricPvalue'])

if metric=="auroc":
	if   roc_frame == "probas":
		idx_p1 = 0;
	elif roc_frame == "scores":
		idx_p1 = 1;
	elif roc_frame == "mean_probas":
		idx_p1 = 2;
	elif roc_frame == "mean_scores":
		idx_p1 = 3;
	#endif
elif metric[0:4]=="mean":
	idx_p1 = 3;
	metric = metric.split('_')[-1];
else:
	idx_p1 = 0;
#endif

for nf in range(1,501):
	empirical = filepath + '/results/class.two.stage.rfe.'+str(nf)+'.empirical/'+metric+'.txt';
	if os.path.isfile(empirical):
		if os.stat(empirical).st_size != 0:
			nf_summ = pd.read_csv(empirical,sep='\t',header=None,index_col=0);
			summary_pnl.loc[nf,'AvgEmpMetric']     =      np.mean(nf_summ.iloc[0,idx_p1])
			summary_pnl.loc[nf,'NumEmpIterations'] = len(np.where(nf_summ.iloc[:,idx_p1])[0]); 
			empPipeOne = nf_summ.iloc[0,idx_p1];
		#endif
	#endif

	if metric=="auroc":

		n_bootstraps = 100;
		BS_auc_, BS_acc_, BS_mcc_ = [[] for ii in range(3)];
		
		trues_dict, scores_dict, probas_dict = grabClassModelFitFromPickle(filepath+'/results/class.two.stage.rfe.'+str(nf)+'.empirical/slurm.log/itr.0.pickle');
		y_true_df, y_pred_df, y_fit_df       = bootstratpClassifierFit(y_true,y_pred,y_score,n_bootstraps,True)	 
		
		for i in range(n_bootstraps):
			BootstrappedClassifierPerformance = ClassifierPerformance(y_true_df.iloc[:,i], y_pred_df.iloc[:,i], y_fit_df.iloc[:,i]);
			BS_auc_.append(BootstrappedClassifierPerformance.Metrics().auc)
			#BS_acc_.append(BootstrappedClassifierPerformance.Metrics().acc)	
			#BS_mcc_.append(BootstrappedClassifierPerformance.Metrics().mcc)	
	
		bootstrappedConfidenceIntervals(BS_auc_,0.05);

	permutation = filepath + '/results/class.two.stage.rfe.'+str(nf)+'.permutations/'+metric+'.txt';
	if os.path.isfile(permutation):
		if os.stat(permutation).st_size != 0:
			nf_summ = pd.read_csv(permutation,sep='\t',header=None,index_col=0);
			summary_pnl.loc[nf,'AvgNullMetric']     =      np.mean(nf_summ.iloc[:,idx_p1])
			summary_pnl.loc[nf,'NumNullIterations'] = len(np.where(nf_summ.iloc[:,idx_p1])[0]);
			summary_pnl.loc[nf,'EmpMetricPvalue']   = float(len(np.where(nf_summ.iloc[:,idx_p1]>empPipeOne)[0]));
			summary_pnl.loc[nf,'EmpMetricPvalue']  /=      (len(np.where(nf_summ.iloc[:,idx_p1])[0])+1);
		#endif
	#endif
	
summary_pnl.to_csv(filepath+'/summary/class.two.stage.rfe.summary.'+metric+'.txt',sep='\t',header=True,index_col=True)


