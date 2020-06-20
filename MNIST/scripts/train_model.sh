cd ..
gpu_id=1
lr_max=1e-3
MODEL_TYPE=$1
model_id=$2

for randomize in {0,1};
do
	for alpha_l_inf in {0.01,0.02,0.03};
	do
		for alpha_l_2 in {0.1,0.2};
		do
			for alpha_l_1 in {0.75,0.8,1.0,2.0};
			do
			
			echo $model_id $gpu_id 
			python train.py \
						-gpu_id $gpu_id \
						-model_id $model_id \
						-model $MODEL_TYPE \
						-alpha_l_1 $alpha_l_1 \
						-alpha_l_2 $alpha_l_2 \
						-alpha_l_inf $alpha_l_inf \
						-lr_max $lr_max \
						-randomize $randomize \
						-k_map 2

			((model_id=model_id+1))
			
			done
		done
	done
done
