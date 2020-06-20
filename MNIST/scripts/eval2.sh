cd ..
gpu_id=$1
MODEL_TYPE=$2
START=$3
END=48

for (( model_id=$START; c<$END; model_id++ ))
do			
	echo $model_id $gpu_id 
done

