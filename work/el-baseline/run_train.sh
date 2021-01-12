export DATA_DIR=../../data/data60899/
export TASK_NAME=other_mention

python ./src/train.py \
  --model_name_or_path ernie-1.0 \
  --data_dir $DATA_DIR \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --output_dir ./output/$TASK_NAME/ \
  --save_steps 5000 \
  --max_steps 20000 \
  --eval_num 1500 \
  --log_dir ./log \
  --other_mention \

python ./src/train.py --model_name_or_path ./ernie/model-ernie1.0.1 --data_dir ../../data/data60899/ --per_device_train_batch_size 32 --learning_rate 5e-5 --output_dir ./output/other_mention/ --save_steps 5000 --max_steps 20000 --eval_num 1500 --log_dir ./log --other_mention