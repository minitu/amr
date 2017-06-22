#!/bin/bash

#PBS -A csc103
#PBS -N AMR-256-CUDA
#PBS -j oe
#PBS -l walltime=02:00:00,nodes=1

# Parameters to be set manually
iterations=10
rundir=/lustre/atlas/proj-shared/csc103/jchoi/amr

execname="advection-cuda"
max_depth=4
num_iters=1
lb_freq=3

logdir="$rundir/results/titan-$PBS_NUM_NODES"
rm -rf $logfile
rm -rf $resultfile

cd $rundir

for array_dim in 256
do
  for block_size in 32
  do
    outfile="$execname-$block_size-$array_dim.out"
    logfile="$execname-$block_size-$array_dim.log"
    resultfile="$execname-$block_size-$array_dim.result"

    runstr="Running $execname $max_depth $block_size $num_iters $lb_freq $array_dim"
    echo $runstr
    echo $runstr >> "$logdir/$resultfile"

    for pVal in 1 2 4 8 16
    do
      echo "Using $pVal PEs"
      run_time_acc=0.0
      decision_time_acc=0.0
      compute_time_acc=0.0
      for (( iteration=1; iteration<=$iterations; iteration++ ))
      do
        echo "Iteration $iteration"

        aprun -n $PBS_NUM_NODES -N 1 ./$execname $max_depth $block_size $num_iters $lb_freq $array_dim ++ppn $pVal > $logdir/$outfile

        run_time=`grep -i "simulation time" $logdir/$outfile | cut -d ":" -f 2 | cut -d " " -f 2`
        decision_time=`grep -i "decision average time" $logdir/$outfile | cut -d ":" -f 2 | cut -d " " -f 2`
        compute_time=`grep -i "function average time" $logdir/$outfile | cut -d ":" -f 2 | cut -d " " -f 2`
        echo "+p $pVal $run_time $decision_time $compute_time" >> $logdir/$logfile
        run_time_acc=$(echo "$run_time_acc + $run_time" | bc)
        decision_time_acc=$(echo "$decision_time_acc + $decision_time" | bc)
        compute_time_acc=$(echo "$compute_time_acc + $compute_time" | bc)
      done
      run_time_avg=$(echo "scale=6; $run_time_acc / $iterations" | bc)
      decision_time_avg=$(echo "scale=6; $decision_time_acc / $iterations" | bc)
      compute_time_avg=$(echo "scale=6; $compute_time_acc / $iterations" | bc)
      echo "[$pVal PEs] average execution time: $run_time_avg, average refinement decision time: $decision_time_avg, average compute function time: $compute_time_avg" >> "$logdir/$resultfile"
    done
  done
done
