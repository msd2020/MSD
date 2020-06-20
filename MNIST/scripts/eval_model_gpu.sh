cd ..
gpu_id=$1
MODEL_TYPE=$2
START=$3
END=48

for (( model_id=$START; c<$END; model_id++ ))
do			
	echo $model_id $gpu_id 
	python evaluate_grid_search.py \
				-gpu_id $gpu_id \
				-batch_size 1000 \
				-res 10 \
				-path Models/$MODEL_TYPE/model_$model_id/iter_15
done

