
i=$1

for a_inf in {0.002,0.003,0.004};
do 
	for a_1 in {1,1.5};
	do
		for lr_max in {0.1,0.2,0.5};
		do
			echo $i
			((i=i+1))
			sbatch job.sh $i $a_inf $a_1 10 $lr_max 1
		done
	done
done

