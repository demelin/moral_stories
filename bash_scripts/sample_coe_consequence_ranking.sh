#!/bin/sh

python3 ./../experiments/run_coe_consequence_ranking_experiment.py \
--consequence_generator_model_type t5 \
--consequence_classifier_model_type roberta \
--consequence_generator_checkpoint None \
--consequence_classifier_checkpoint None \
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
--num_consequences 10
