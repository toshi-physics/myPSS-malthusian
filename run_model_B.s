#!/bin/bash

source ~/miniconda3/bin/activate pss-env

#set -x

sh_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

src_dir="$(realpath "${sh_dir}/src")"

data_dir="$(realpath "${sh_dir}/data")"

if (( $# != 3 )); then
    echo "Usage: run_model.s model_name rhoend run"
    exit 1
fi

model=$1
rhoend=$(python3 -c "print('{:.3f}'.format($2))")
run=$(python3 -c "print('{:d}'.format($3))")

T=2
n_steps=1e+4
dt_dump=0.1
mx=100
my=100
dx=1.0
dy=1.0

a=50
d=5
k_g=1
rhoseed=0.1
rhomax=2.0
rhomin=0.05


save_dir="${sh_dir}/data/$model/rhoend_${rhoend}/run_${run}"

if [ ! -d $save_dir ]; then
    mkdir -p $save_dir
fi

params_file="${save_dir}/parameters.json"

echo \
"
{
    "\"run\"" : $run,
    "\"T\"" : $T,
    "\"n_steps\"" : $n_steps,
    "\"dt_dump\"" : $dt_dump,
    "\"k_g\"" : $k_g,
    "\"a\"": $a,
    "\"d\"": $d,
    "\"rhoend\"": $rhoend,
    "\"rhoseed\"" : $rhoseed,
    "\"rhomin\"" : $rhomin,
    "\"rhomax\"" : $rhomax,
    "\"mx\"" : $mx,
    "\"my\"" : $my,
    "\"dx\"" : $dx,
    "\"dy\"" : $dy
}
" > $params_file

python3 -m models.$model -s $save_dir
python3 -m src.analysis.create_videos_rho -s $save_dir