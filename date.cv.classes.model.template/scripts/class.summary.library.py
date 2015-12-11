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


