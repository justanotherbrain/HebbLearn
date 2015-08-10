qsub -I -X -l nodes=1:ppn=2 -l walltime=100:00:00 -l mem=120GB
module load scipy
module rm numpy
module load h5py
