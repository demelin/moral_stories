# A summary for flags for run_baseline_experiments.py

## Required Arguments
* <code>--data_dir</code>: The input data dir. Should contain the .tsv files (or other data files) for the task. For more, refer to <code>/bash_scripts</code>

* <code>--model_type</code>: Model type from the TASK_DICT. In the paper, we use 'roberta', 'gpt2', 'bart' and 't5', but this list can be extended by adding it in the script.

* <code>--model_name_or_path</code>: Path to pre-trained model or model name from Huggingface hub.
 
* <code>--task_name</code>: The name of the task to train for e.g. `action_cls`. For more, refer to the script or `/bash_scripts`.

* <code>--split_name</code>: The name of the data split used to train / evaluate the model. This can be one of `norm_distance`, `minimal_pairs` or `lexical_bias`.

* <code>--output_dir</code>: The root output directory where the model predictions and checkpoints will be written.

## Train or Eval?

* <code>--do_train</code>: Whether to run training.

* <code>--do_eval</code>: Whether to run eval on the dev set.

* <code>--do_prediction</code>: Whether to run prediction on the test set. (Training will not be executed.)

* <code>--evaluate_during_training</code>: Do evaluation during training at each logging step.

## Optional Parameters

### Training parameters

* <code>--learning_rate</code>: The initial learning rate for Adam. (default=5e-5)
* <code>--weight_decay</code>: Weight deay if we apply some. (default=0.0)

* <code>--adam_epsilon</code>: Epsilon for Adam optimizer. (default=1e-8)

* <code>--max_grad_norm</code>: Max gradient norm. default=1.0

* <code>--num_train_epochs</code>: Total number of training epochs to perform. (default=3.0)

* <code>--max_steps</code>: If > 0: set total number of training steps to perform. Override num_train_epochs. (default=-1)

* <code>--warmup_steps</code>: Linear warmup over warmup_steps. (default=0)

* <code>--warmup_pct</code>: Linear warmup over warmup_pct * total_steps.

* <code>--seed</code>: random seed for initialization (default=42)


### Generation parameters

* <code>--max_gen_length</code>: The maximum length of the sequence to be generated. (default = 60)

* <code>--temperature</code>: The value used to module the next token probabilities. (default=1.0)

* <code>--k</code>: The number of highest probability vocabulary tokens to keep for top-k-filtering. (default=0)

* <code>--p</code>: If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. (default=0)
* <code>--num_beams</code>: number of beams for beam search (default=0)

* <code>--do_sample</code>: Whether to generate predictions via sampling; if off, decoding is done greedily. (Add flag if wanted)

* <code>--patience</code>: Number of epochs without dev-set loss improvement to ignore before stopping the training. (default=-1)

### Other logistical parameters

* <code>--config_name</code>: Pretrained config name or path if not the same as model_name.

* <code>--tokenizer_name</code>: Pretrained tokenizer name or path if not the same as model_name

* <code>--cache_dir</code>: The cache directory where do you want to store the pre-trained models downloaded from s3

* <code>--max_seq_length</code>: The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. (default=128)

* <code>--do_lower_case</code>: Set this flag if you are using an uncased model.

* <code>--run_on_test</code>: Evaluate model on the test split.

* <code>--data_cache_dir</code>: The root directory for caching features.

* <code>--pretrained_dir</code>: The directory containing the checkpoint of a pretrained model to be fine-tuned.

* <code>--save_total_limit</code>: Maximum number of checkpoints to keep

* <code>--per_gpu_train_batch_size</code>: Batch size per GPU/CPU for training. (default=8)

* <code>--per_gpu_eval_batch_size</code>: Batch size per GPU/CPU for evaluation. (default=8)

* <code>--gradient_accumulation_steps</code>: Number of updates steps to accumulate before performing a backward/update pass.

* <code>--logging_steps</code>: Log every X updates steps. (default=50)

* <code>--save_steps</code>: Save checkpoint every X updates steps. (default=50)

* <code>--eval_all_checkpoints</code>: Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number

* <code>--no_cuda</code>: Avoid using CUDA when available

* <code>--overwrite_output_dir</code>: Overwrite the content of the output directory

* <code>--overwrite_cache</code>: Overwrite the cached training and evaluation sets

* <code>--fp16</code>: Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit

* <code>--fp16_opt_level</code>: help='For fp16: Apex AMP optimization level selected in [\'O0\', \'O1\', \'O2\', and \'O3\']. See details at https://nvidia.github.io/apex/amp.html') (default='O1')

* <code>--local_rank</code>: For distributed training: local_rank (default=-1)

* <code>--server_ip</code>: For distant debugging.

* <code>--server_port</code>: For distant debugging.