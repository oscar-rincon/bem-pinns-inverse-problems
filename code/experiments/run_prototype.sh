#!/bin/sh
# Experiment 9 (amplitude-only data): classical solvers in sequence, BEM-PINN four at a time.
cd "$(dirname "$0")"
export PYTHONPATH=.. NTHREADS=4
L=../results/logs
mkdir -p $L
for c in interface surface joint; do for n in 0 0.01 0.03; do
  for m in lm_fd lbfgs_adj lbfgs_adj_cont; do
    python3 -W ignore exp9_prototype.py $c $n $m >> $L/exp9_classical.log 2>&1
  done
done; done
(for c in interface surface joint; do for n in 0 0.01 0.03; do echo "$c $n"; done; done) |
  xargs -P 4 -L 1 sh -c 'python3 -W ignore exp9_prototype.py $0 $1 bem_pinn > ../results/logs/exp9_pinn_$0_$1.log 2>&1'
echo done
