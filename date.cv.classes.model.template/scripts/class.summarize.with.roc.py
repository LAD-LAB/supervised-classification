#!/usr/env/bin python

# ___author___ :Firas Said Midani
# ___e-mail___ :firas.midani@duke.edu
# ___date_____ :2015.12.08
# ___version__ :1.0

# TO IMPROE

# ENABLE FOR ACCURACY AND MCC METRICS
# HOW DO YOU DEFINE TEH ITERATIONS? SOMETIMES A MODEL RUN SKIPS CERTAIN NUBERS
# IF AUROC, SELECT WHETEHR YOU WANT TO USE PROBABILITIES OR SCORES 
# ADD ARGUMENT FOR PERCENTILE

########################################################################
## DESCRIPTION
########################################################################

# bootstrapping code is derived from http://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals
# the results of this script corresponds with results by pROC package in R. 

########################################################################
## INITIALISATION
########################################################################

import \
pandas as pd, \
numpy  as np, \
pickle, \
sys

from sklearn.metrics import roc_auc_score

########################################################################
## INPUT HANDLING
########################################################################

filepath=sys.argv[1]

def model_predictions_from_pickle(PicklePath):	
	# Given the path to a pickle file, open it and extract the classifier predictions.
	# Specifically, extract the true labels, the probabilities, and distances from hyperplane for each sample	

	pfid = open(PicklePath,'rb');
	
	iJar = pickle.load(pfid); # items in Jar
	pJar = pickle.load(pfid); # pickles in Jar
	
	cv_tests_ix,cv_trues,cv_scores,cv_probas = [np.where([i==varb for i in iJar])[0][0] for varb in ["cv_tests_ix","cv_trues","cv_scores","cv_probas"]]
	 
	trues  = [(x,y) for x,y in zip(pJar[cv_tests_ix],pJar[cv_trues])];
	scores = [(x,y) for x,y in zip(pJar[cv_tests_ix],pJar[cv_scores])];
	probas = [(x,y) for x,y in zip(pJar[cv_tests_ix],pJar[cv_probas])];

	trues_dict, scores_dict, probas_dict = [{} for ii in range(3)];
	
	trues_dict.update([ii for ii in trues]);
	scores_dict.update([ii for ii in scores]);
	probas_dict.update([ii for ii in probas]);

	# returned items are dictionaries
	return trues_dict, scores_dict, probas_dict	

# for each iteration grab probabilities into a data frame of samples by iteration
trues_df, scores_df, probas_df = [pd.DataFrame() for ii in range(3)];
for itr in range(50):
	pickle_path = 'itr.'+str(itr)+'.pickle';
	trues, scores, probas                       = model_predictions_from_pickle(filepath+'/'+pickle_path);
	trues_df[itr],scores_df[itr],probas_df[itr] = [pd.Series(ii,name=itr) for ii in [trues,scores,probas]]
	#each df has size (num of samples) by (number of model iterations)

# given dataframes of model predictions by all of its iterations,
#  summarize the predicion of each sample as a mean value
list_dict                  = [trues_df,scores_df,probas_df];
list_labels                = ['trues','mean_scores','mean_probas']; 
y_true, y_pred_s, y_pred_p = [pd.Series(x.apply(np.mean,1),name=y) for x,y in zip(list_dict,list_labels)]

# save the summary (averaged) predictions
mean_predictions_df= pd.DataFrame([y_true,y_pred_s,y_pred_p]).T;
mean_predictions_df.to_csv(filepath+'/model_predictions_df.txt',sep='\t',header=True,index_col=True)

########################################################################
## BOOTSTRAPING FOR COMPUTING CONFIDENCE INTERVALS
########################################################################
n_bootstraps          = 1000;
rng_seed              = 6006;
bootstrapped_scores   = [];
 
y_pred_s  = y_pred_s.values
y_pred_p  = y_pred_p.values
y_true    = y_true.values

# for each bootstrap, randomly sample with replacement, and recompute Area under ROC (AuROC)
rng = np.random.RandomState(rng_seed);
for i in range(n_bootstraps):
	indices = rng.random_integers(0,len(y_pred_s)-1, len(y_pred_s));
	if len(np.unique(y_true[indices])) < 2:
		continue
	bootstrapped_scores.append(roc_auc_score(y_true[indices],y_pred_s[indices]));

sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()

# confidence intervals are simply the percentiles of the sorted bootstrapped AuROCs
confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

print("Confidence interval for the score: ["+("%0.4f" % confidence_lower)+" - "+("%0.4f" % confidence_upper)+"]")
print ("Orginal ROC area: "+("%0.3f" % (roc_auc_score(y_true,y_pred_s))))
