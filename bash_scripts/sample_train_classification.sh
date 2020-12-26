#!/bin/sh

python3 ./../experiments/run_baseline_experiment.py \
--model_type roberta \
--model_name_or_path roberta-large \
--task_name action_cls \
--split_name norm_distance \
--do_eval \
--do_prediction \
--do_lower_case \
--data_dir ./../../data \
--max_seq_length 100 \
--per_gpu_eval_batch_size 16 \
--per_gpu_train_batch_size 16 \
--learning_rate 1e-5 \
--num_train_epochs 50 \
--output_dir ./../../output \
--do_train \
--logging_steps 500 \
--save_steps 500 \
--seed 42 \
--data_cache_dir ./../../cache \
--warmup_pct 0.1 \
--evaluate_during_training \
--save_total_limit 10 \
--patience 10 \
--overwrite_output_dir