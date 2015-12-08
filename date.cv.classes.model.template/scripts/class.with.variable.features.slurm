#!/bin/sh

#SBATCH --time-min=1200
#SBATCH --mem=4096MB

source /home/lad44/davidlab/users/fsm/cholera/virtual_python_cholera/bin/activate

prefix=$1
simname=$(basename $prefix)
pypath=$prefix/scripts
simpath=$prefix

txt_out=$simpath/$simname.driver.out
txt_err=$simpath/$simname.driver.err

srun -o $txt_out -e $txt_err python $pypath/class.with.variable.features.py $prefix $2 
