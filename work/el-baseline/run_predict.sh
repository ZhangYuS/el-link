export DATA_DIR=../../data/data60899/
export TASK_NAME=other_mention

python ./src/predict.py \
  --model_name_or_path ernie-1.0 \
  --data_dir $DATA_DIR \
  --per_device_eval_batch_size 36 \
  --output_dir ./output/$TASK_NAME/ \
  --load_from output/$TASK_NAME/step_20000 \
  --mode test \
  --other_mention
