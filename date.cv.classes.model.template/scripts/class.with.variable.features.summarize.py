#!/usr/bin/env python

# __author__ = Firas Said Midani
# __e-mail__ = firas.midani@duke.edu
# __update__ = 2015.12.07
# __version_ = 1.0

import \
pandas as pd, \
numpy  as np, \
time, \
sys, \
os

#filepath='/home/lad44/davidlab/users/fsm/cholera/20151130.HoldoutValidation.YGBR.ridge';
filepath  = sys.argv[1]
metric    = sys.argv[2]
if len(sys.argv)>3:
	roc_frame = sys.argv[3]
#endif

summary_pnl = pd.DataFrame(columns=['PipeOneEmp','PipeOnePerm','PipeOnePermNum','PipeOnePvalue','PipeTwoEmp','PipeTwoPerm','PipeTwoPermNum','PipeTwoPvalue'])

if metric=="auroc":
	if   roc_frame == "probas":
		idx_p1 = 0;
		idx_p2 = 4	
	elif roc_frame == "scores":
		idx_p1 = 1;
		idx_p2 = 5
	elif roc_frame == "mean_probas":
		idx_p1 = 6;
		idx_p2 = 4
	elif roc_frame == "mean_scores":
		idx_p1 = 7;
		idx_p2 = 5;
	#endif
elif metric[0:4]=="mean":
	idx_p1 = 3;
	idx_p2 = 2;
	metric = metric.split('_')[-1];
else:
	idx_p1 = 0;
	idx_p2 = 2;
#endif

for nf in range(1,501):
	empirical = filepath + '/results/class.two.stage.rfe.'+str(nf)+'.empirical/'+metric+'.txt';
	if os.path.isfile(empirical):
		if os.stat(empirical).st_size != 0:
			nf_summ = pd.read_csv(empirical,sep='\t',header=None,index_col=0);
			summary_pnl.loc[nf,'PipeOneEmp'] = nf_summ.iloc[0,idx_p1]
			summary_pnl.loc[nf,'PipeTwoEmp'] = nf_summ.iloc[0,idx_p2]
			empPipeOne = nf_summ.iloc[0,idx_p1];
			empPipeTwo = nf_summ.iloc[0,idx_p2];
		#endif
	#endif
	permutation = filepath + '/results/class.two.stage.rfe.'+str(nf)+'.permutations/'+metric+'.txt';
	if os.path.isfile(permutation):
		if os.stat(permutation).st_size != 0:
			nf_summ = pd.read_csv(permutation,sep='\t',header=None,index_col=0);
			summary_pnl.loc[nf,'PipeOnePerm']    = np.mean(nf_summ.iloc[:,idx_p1])
			summary_pnl.loc[nf,'PipeTwoPerm']    = np.mean(nf_summ.iloc[:,idx_p2])
			summary_pnl.loc[nf,'PipeOnePermNum'] = len(np.where(nf_summ.iloc[:,idx_p1])[0]);
			summary_pnl.loc[nf,'PipeTwoPermNum'] = len(np.where(nf_summ.iloc[:,idx_p2])[0]);
			summary_pnl.loc[nf,'PipeOnePvalue']  = float(len(np.where(nf_summ.iloc[:,idx_p1]>empPipeOne)[0]))/(len(np.where(nf_summ.iloc[:,idx_p1])[0])+1);
			summary_pnl.loc[nf,'PipeTwoPvalue']  = float(len(np.where(nf_summ.iloc[:,idx_p2]>empPipeTwo)[0]))/(len(np.where(nf_summ.iloc[:,idx_p2])[0])+1);
		#endif
	#endif

summary_pnl.to_csv(filepath+'/summary/class.two.stage.rfe.summary.'+metric+'.txt',sep='\t',header=True,index_col=True)


