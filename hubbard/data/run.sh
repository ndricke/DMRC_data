#!/bin/bash


run_path=$PWD
exe="$run_path/../script/gen_nunif_hub_data.py"

nsample=100

for nsite in 10 12
do
	nocc_max=`echo $nsite | awk '{print $1/2}'`
	for nocc in `seq 1 1 $nocc_max`
	do
		for sigma in 0.1 0.3 0.5 1.0
		do
			for U in 1 4 8
			do
				outpath=n${nsite}_no${nocc}_s${sigma}_U${U}

				mkdir -p $outpath
				cd $outpath

				cat > submit.sh << eof
#!/bin/sh

#SBATCH -t 100:00:00
#SBATCH -o sbatch.out
#SBATCH -e sbatch.err
#SBATCH -c 1
#SBATCH -N 1
#SBATCH --mem=2000
#SBATCH -x n[02-03]

# export lib path for user specific lib
export PYTHONPATH=\${PYTHONPATH}:/home/hzye2011/.local/lib/python2.7/site-packages/pip/_vendor
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/home/mmavros/lib

export OMP_NUM_THREADS=1

python $exe $nsite $nocc $U $sigma $nsample ./
eof

			sbatch -J $outpath submit.sh

			cd .. # $outpath
			done
		done
	done
done
