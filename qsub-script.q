#!/bin/bash
#PBS -l nodes=1:ppn=1,walltime=10:00:00,mem=120GB
#PBS -N hebblearn
#PBS -M mad573@nyu.edu
#PBS -m abe
#PBS -e localhost:/scratch/mad573/${PBS_JOBNAME}.e${PBS_JOBID}
#PBS -o localhost:/scratch/mad573/${PBS_JOBNAME}.o${PBS_JOBID}
module load scipy
cd ~/research/hebblearn
#python demo.py 6 3 10 1 100000
python multilayer-demo.py
exit 0;
