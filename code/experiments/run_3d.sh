#!/bin/sh
# 3D experiments (after experiment 5 has finished). 4 threads per process.
cd "$(dirname "$0")"
export PYTHONPATH=.. NTHREADS=4
L=../results/logs
while [ ! -f ../results/exp5_320_bem_pinn.json ] || [ ! -f ../results/exp5_160_bem_pinn.json ]; do sleep 30; done
python3 -W ignore exp7_scaling3d.py > $L/exp7.log 2>&1
for n in 0 0.01 0.03; do for m in lbfgs_lu lm_lu lbfgs_gmres; do
  python3 -W ignore exp6_3d.py $n $m >> $L/exp6_classical.log 2>&1
done; done
(for n in 0 0.01 0.03; do echo $n; done) | xargs -P 3 -I{} sh -c 'python3 -W ignore exp6_3d.py {} bem_pinn > ../results/logs/exp6_pinn_{}.log 2>&1'
echo done
