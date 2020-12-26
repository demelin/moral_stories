#!/bin/sh

python3 ./../experiments/run_coe_action_ranking_experiment.py \
--action_draft_model_type bart \
--action_classifier_model_type roberta \
--action_generator_checkpoint None \
--action_classifier_checkpoint None \
--split_name norm_distance \
--do_lower_case \
--do_sample \
--data_dir ./../../data \
--max_seq_length 100 \
--per_gpu_eval_batch_size 8 \
--output_dir ./../../output \
--seed 42 \
--data_cache_dir ./../../cache \
--max_gen_length 60 \
--p 0.9 \
--num_actions 10