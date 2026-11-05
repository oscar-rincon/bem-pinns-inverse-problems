#!/bin/sh
# Experiment 8 (basin of attraction): classical solvers in sequence, BEM-PINN four at a time.
cd "$(dirname "$0")"
export PYTHONPATH=.. NTHREADS=4
L=../results/logs
mkdir -p $L
for b in low mid high; do for h in 0.1 0.2 0.35 0.5 0.9 1.1 1.3 1.5; do
  for m in lm_fd lbfgs_adj lbfgs_adj_cont; do
    python3 -W ignore exp8_basin.py $b $h $m >> $L/exp8_classical.log 2>&1
  done
done; done
(for b in low mid high; do for h in 0.1 0.2 0.35 0.5 0.9 1.1 1.3 1.5; do echo "$b $h 0"; done; done
 for s in 1 2; do for c in "mid 1.3" "mid 1.5" "high 0.2" "high 1.1"; do echo "$c $s"; done; done) |
  xargs -P 4 -L 1 sh -c 'python3 -W ignore exp8_basin.py $0 $1 bem_pinn $2 > ../results/logs/exp8_pinn_$0_$1_s$2.log 2>&1'
echo done
