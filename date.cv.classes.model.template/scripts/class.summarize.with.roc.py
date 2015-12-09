#!/usr/env/bin python

# ___author___ :Firas Said Midani
# ___e-mail___ :firas.midani@duke.edu
# ___date_____ :2015.12.08
# ___version__ :1.0

########################################################################
## DESCRIPTION
########################################################################


########################################################################
## INITIALISATIoN
########################################################################

import \
pandas as pd, \
numpy  as np, \
pickle, \
sys

filepath=sys.argv[1]


def dictionary_from_pickle(PicklePath):
	pfid = open(PicklePath,'rb');
	
	iJar = pickle.load(pfid); # items in Jar
	pJar = pickle.load(pfid); # pickles in Jar
	
	df   = pd.DataFrame(index=range(77),columns=["sample_index","true_label","prob(1)"]);
	
	cv_tests_ix,cv_trues,cv_scores = [np.where([i==varb for i in iJar])[0] for varb in ["cv_tests_ix","cv_trues","cv_scores"]]
	
	trues  = [(x,y) for x,y in zip(pJar[cv_tests_ix],pJar[cv_trues])];
	probas = [(x,y) for x,y in zip(pJar[cv_tests_ix],pJar[cv_scores])];
	
	trues_dict, probas_dict = [{} for ii in range(2)];
	
	trues_dict.update([ii for ii in trues])
	probas_dict.update([ii for ii in probas]);

	return trues_dict, probas_dict	

# for each iteration grab probabilities into a data frame of samples by iteration

trues_dict,probas_dict = [pd.DataFrame() for ii in range(2)];
for p in range(5):
	pickle_path = 'itr.'+str(p)+'.pickle';
	trues, probas  = dictionary_from_pickle(filepath+'/'+pickle_path);
	trues_dict[p]  = pd.Series(trues,name=p)
	probas_dict[p] = pd.Series(probas,name=p)	

trues_series  = pd.Series(trues_dict.apply(np.mean,1),name='trues')
probas_series = pd.Series(probas_dict.apply(np.mean,1),name='mean_scores')
mean_df       = pd.DataFrame([trues_series,probas_series]).T;
mean_df.to_csv(filepath+'/mean_scores_df.txt',sep='\t',header=True,index_col=True) 
