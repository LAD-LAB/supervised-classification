#!/usr/bin/env python

import sys, os 
import time

filepath=sys.argv[1];

# each iteraction performs pipelines 1 and 2 on select number of features
#      include permutation tests to pipelines

cnt=0;
for nf in [1]+range(10,201,10):
	# modify a copy of parameter file 
	with open(filepath+'/scripts/class.rfe.modelparameters.py') as f:
		new_file = open(filepath+'/results/param.files/class.rfe.'+str(nf)+'.model.parameters.py','w');
		for line in f:
			if line.startswith('num_features_2'):
				new_file.write('num_features_2   = %s\n' % str(nf));
			#elif line.startswith('#SBATCH --job-name'):
			#	new_file.write(line.rstrip()+'.empirical\n');
			else:
				new_file.write(line)
		new_file.close()

	# modify a copy of job file (EMPIRICAL)
	with open(filepath+'/scripts/class.two.stage.rfe.template.slurm') as f:
		new_file = open(filepath+'/results/slurm.filesclass.two.stage.rfe.'+str(nf)+'.empirical.slurm','w');
		for line in f:
			if   line.startswith('prefix'):
				new_file.write('prefix='+filepath+'\n');
			elif line.startswith('numperm'):
				new_file.write('numperm=1\n');
			elif line.startswith('shuffle'):
				new_file.write('shuffle=0\n');
			elif line.startswith('pickle_model'):
				new_file.write('pickle_model=1\n');
			elif line.startswith('txt_params'):
				new_file.write('txt_params=$prefix/results/param.files/class.rfe.%s.model.parameters.py\n' % str(nf));
			elif line.startswith('simname'):
				new_file.write('simname=class.rfe.two.stage.rfe.'+str(nf)+'.empirical\n');
				#new_file.write('simname='+filepath.split("/")[-1]+'\n');
			#elif line.startswith('#SBATCH --job-name'):
			#	new_file.write(line.rstrip()+'.permutation\n');
			else:
				new_file.write(line)
		new_file.close()
		os.system('sbatch '+filepath+'/results/slurm.files.class.rfe.two.stage.rfe.'+str(nf)+'.empirical.slurm');

	# modify a copy of job file (PERMUTATIONS)
#	with open(filepath+'/scripts/class.two.stage.rfe.template.slurm') as f:
#		new_file = open(filepath+'/results/slurm.files/class.two.stage.rfe.'+str(nf)+'.permutations.slurm','w');
#		for line in f:
#			if line.startswith('numperm'):
#				new_file.write('numperm=500\n');
#			elif line.startswith('shuffle'):
#				new_file.write('shuffle=1\n');
#                       elif line.startswith('pickle_model');
#                               new_file.write('pickle_model=0\n');
#			elif line.startswith('txt_params'):
#				new_file.write('txt_params=$prefix/results/param.files/class.rfe.%s.model.parameters.py\n' % str(nf));
#			elif line.startswith('simname'):
#				new_file.write('simname=two.stage.rfe.'+str(nf)+'.permutations\n');
#			else:
#				new_file.write(line)
#		new_file.close()
#		os.system('sbatch '+filepath+'/results/slurm.files/class.two.stage.rfe.'+str(nf)+'.permutations.slurm');
	cnt+=1;
	#if cnt>3:
	time.sleep(1)
	
 
