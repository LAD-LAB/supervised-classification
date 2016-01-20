#!/bin/sh

#SBATCH --time-min=120
#SBATCH --mem=1024MB

source /home/lad44/davidlab/users/fsm/cholera/vpython/bin/activate

prefix=
simname=
pypath=$prefix/scripts
simpath=$prefix/results/$simname

if [ ! -d "$simpath" ]; then mkdir "$simpath" ; fi

txt_out=$simpath/$simname.out
txt_err=$simpath/$simname.err

txt_params=$prefix/scripts/class.two.stage.rfe.model.parameters.py
txt_features_df=$prefix/data/features_df.txt
txt_taxa_level_map=$prefix/data/taxa_level_map.txt
txt_mapping=$prefix/data/smic.study.mapping.txt
txt_clinical=$prefix/data/smic.study.clinical.data.txt
num_features=
numperm=100
shuffle=1
pickle_model=0

srun -o $txt_out -e $txt_err python $pypath/class.multiple.versatile.py $txt_params $txt_features_df $txt_taxa_level_map  $txt_mapping $txt_clinical $simpath $simname $shuffle $numperm $num_features $pickle_model
