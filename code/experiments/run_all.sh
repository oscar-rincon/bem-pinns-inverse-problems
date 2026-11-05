#!/bin/sh
# Runs experiments 3, 2 and 4 after experiment 1 (see README). 4 threads per process.
cd "$(dirname "$0")"
export PYTHONPATH=.. NTHREADS=4
L=../results/logs
mkdir -p $L
python3 -W ignore exp3_scaling.py > $L/exp3.log 2>&1
for c in interface surface joint; do for n in 0 0.01 0.03; do for m in lm_fd lbfgs_adj; do
  python3 -W ignore exp2_inverse.py $c $n $m >> $L/exp2_classical.log 2>&1
done; done; done
python3 -W ignore exp4_frequency.py > $L/exp4.log 2>&1 &
(for c in interface surface joint; do for n in 0 0.01 0.03; do echo "$c $n"; done; done) |
  xargs -P 3 -L 1 sh -c 'python3 -W ignore exp2_inverse.py $0 $1 bem_pinn > ../results/logs/exp2_pinn_$0_$1.log 2>&1'
wait
echo done
