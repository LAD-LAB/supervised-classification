#!/usr/bin/env python -W ignore::DeprecationWarning
#
# ___author___ : Firas Said Midani
# ___e-mail___ : firas.midani@duke.edu
# ___date_____ : 2015.12.11
# ___version__ : 1.0
#
# LIST OF FUNCTIONS
#
# grabClassModelFitFromPickle
# 	given a path to a pickle file, extract model fit/predictions
#       and return as dictionaries of sample index vs fit/predictions
#

import warnings
warnings.filterwarnings('ignore')

import \
    pandas as pd, \
    numpy as np,\
    pickle, \
    random, \
    time, \
    sys, \
    os
    
#####################################
# classification auxiliary tools
#####################################

from sklearn.metrics            import auc, roc_curve, roc_auc_score, accuracy_score, matthews_corrcoef

#####################################

def grabClassModelFitFromPickle(PicklePath):
	# Given the path to a pickle file, open it and extract the classifier predictions.
	# Specifically, extract the true labels, the probabilities, and the distances from the hyperplane for each sample. 

	pfid = open(PicklePath,'rb');

	iJar = pickle.load(pfid); # items in Jar
	pJar - pickle.load(pfid); # pickles in Jar

	cv_tests_ix,cv_trues,cv_scores,cv_probas = [np.where([i==varb for i in iJar])[0][0] for varb in ["cv_tests_ix","cv_trues","cv_scores","cv_probas"]];

	trues  = [(x,y) for x,y in zip(pJar[cv_tests_ix],pJar[cv_trues])];
	scores = [(x,y) for x,y in zip(pJar[cv_tests_ix],pJar[cv_scores])];
	probas = [(x,y) for x,y in zip(pJar[cv_tests_ix],pJar[cv_probas])];

	trues_dict, scores_dict, probas_dict = [{} for ii in range(3)];
	
	trues_dict.update([ii for ii in trues]);
	scores_dict.update([ii for ii in scores]);
	probas_dict.update([ii for ii in probas]);

	#returned items are dictionaries
	return trues_dict, scores_dict, probas_dict

#####################################

class ClassifierPerformance(object):
	"""The performance of a classifier as evaluated with ROC curves and other classification metrics

	Attributes:
		name: A string represeting teh classifiers name
	"""

	def __init__(self,y_true,y_pred,y_fit):
		"""Return a ClassifierPerformance object"""

		self.y_true = y_true;
		self.y_pred = y_pred;
		self.y_fit  = y_fit;

	def roc(self):
		"""Return fpr, tpr, and auc"""

		fpr, tpr, thresholds = roc_curve(self.y_true,self.y_fit)

		self.fpr = fpr;
		self.tpr = tpr;

		return self

	def accuracy_score(self):
		"""Return accuracy score"""

		self.acc = accuracy_score(self.y_true,self.y_pred)
	
		return self

	def matthews_score(self):
		"""Return Matthew's correlation coefficient"""

		self.mcc = matthews_corrcoef(self.y_true,self.y_pred)

		return self

	def Metrics(self):
		"""Return all classifier metrics: auc, accuracy, and matthew's."""

		roc = self.roc();
		acc = self.accuracy_score().acc;
		mcc = self.matthews_score().mcc;

		self.auc = auc(roc.fpr,roc.tpr);
		self.acc = acc;
		self.mcc = mcc;

		return self

def bootstrapClassifierFit(y_true,y_pred,y_fit,n_bootstraps,stratify=True):
	
	y_true_df,y_fit_df,y_pred_df = [pd.DataFrame(index=range(len(y_true)),columns=range(n_bootstraps)) for ii in range(3)];
	
	rng_seed = 6006;	
	rng = np.random.RandomState(rng_seed);
	for i in range(n_bootstraps):
		
		if stratify:
			pos_ix  = np.random.choice(np.where(y_true==1)[0],size=np.sum(y_true==1),replace=True);
			neg_ix  = np.random.choice(np.where(y_true==0)[0],size=np.sum(y_true==0),replace=True);
			indices = np.concatenate([pos_ix,neg_ix]).;
		else:
			indices = np.random_integers(0,len(y_fit)-1,len(y_fit));
	
		if len(np.unique(y_true[indices])) >1:
			y_true_df.loc[:,i] = y_true[indices];
			y_pred_df.loc[:,i] = y_pred[indices];
			y_fit_df.loc[:,i]  = y_fit[indices];

	return y_true_df,y_pred_df,y_fit_df

def bootstrappedConfidenceIntervals(BS_metric,alpha):

	sorted_BS_metric = np.array(BS_metric);
	sorted_BS_metric.sort()

	lowerPercentile = float(alpha)/2;
	upperPercentile = 1-float(alpha)/2;

	confidence_lower = sorted_BS_metric[int(lowerPercentile * len(sorted_BS_metric)];
	confidence_upper = sorted_BS_metric[int(upperPercentile * len(sorted_BS_metric)];

	return confidence_lower, confidence_upper

