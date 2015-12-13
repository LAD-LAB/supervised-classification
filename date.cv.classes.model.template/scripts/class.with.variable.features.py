#!/usr/bin/env python

# ___author___  = Firas Said Midani
# ___e-mail___  = firas.midani@duke.edu
# ___date_____  = 2015.12.07
# ___version__  = 1.0

##############################################################
# DESCRIPTION
##############################################################

# Given template slurm scripts, construct new slurm scripts 
#  and modify the number of features (num_features) argument, 
#  and define the location of the desired model and its name,
#  with the latter two extracted from this file's arguments.

##############################################################
# INPUT
##############################################################

# filepath = sys.argv[1]; path to store model and its results
# permute  = sys.argv[2]; 1="run permuted verisons of the model, 0="don't"


##############################################################
# INITIALIZATION AND INPUT HANDLING
##############################################################i

import time, sys, os

filepath = sys.argv[1];
permute  = int(sys.argv[2]);

##############################################################i
# MISCELLANEOUS
##############################################################i

# each iteration performs pipelines 1 (cross-validation) and 2 (hold-out validation) 
#       on select number of features  include permutation tests to pipelines

##############################################################i
# PRIMARY CODE
##############################################################i

model_iterations = [150];#+range(10,201,10); 
run_type         = ['empirical','permutations'];

for nf in model_iterations:
	for rt in range(0,1+permute):
		# if permute==0, then the for loop will only build slurm scripts for running empirical models
		# if permute=1,  then the for loop will also build slurm scripts for running empirical and  permuted models	i
		run = run_type[rt];
		with open(filepath+'/scripts/class.two.stage.rfe.template.'+run+'.slurm') as f:
			new_file = open(filepath+'/results/slurm.files/class.two.stage.rfe.'+str(nf)+'.'+run+'.slurm','w');
			for line in f:
				if   line.startswith('prefix'):
					new_file.write('prefix='+filepath+'\n');
				elif line.startswith('num_features'):
					new_file.write('num_features=%s\n' % str(nf));
				elif line.startswith('simname'):
					new_file.write('simname=class.two.stage.rfe.'+str(nf)+'.'+run+'\n');
				else:
					new_file.write(line)
			new_file.close()
			os.system('sbatch '+filepath+'/results/slurm.files/class.two.stage.rfe.'+str(nf)+'.'+run+'.slurm');
	time.sleep(900)
	

