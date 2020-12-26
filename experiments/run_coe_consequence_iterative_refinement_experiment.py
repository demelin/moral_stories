from __future__ import absolute_import, division, print_function

import os
import json
import math
import torch
import logging
import argparse

import numpy as np

from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import (DataLoader,
                              RandomSampler,
                              SequentialSampler,
                              DistributedSampler,
                              TensorDataset)

from transformers import (RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          T5Config,
                          T5ForConditionalGeneration,
                          T5Tokenizer,
                          BartConfig,
                          BartForConditionalGeneration,
                          BartTokenizer,
                          GPT2Config,
                          GPT2LMHeadModel,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup)

from utils import (compute_bleu, compute_gen_metrics, convert_examples_to_features, MoralStoriesProcessor, set_seed,
                   get_token_loss, _rotate_checkpoints, compute_cls_metrics)


logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
}

TASKS = ['consequence|action+context_gen', 'consequence+action+context_cls', 'consequence|action+context_genref']

TASK_DICT = {
                'consequence|action+context_gen': ['nrm', 'sit', 'itn', 'ma', 'ia', 'mc', 'ic'],
                'consequence+action+context_cls': ['nrm', 'sit', 'itn', 'ma', 'ia', 'mc', 'ic'],
                'consequence|action+context_genref': ['nrm', 'sit', 'itn', 'ma', 'ia', 'mcd', 'icd', 'mc', 'ic']
}
SPLITS = ['norm_distance', 'lexical_bias', 'minimal_pairs']
MAX_GEN_LENGTH = int(10000)


def train(args, model, tokenizer, task_name, model_type, train_dataset, dev_dataset):
    """ Trains the model. """
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    # Set-up TensorBoard logger
    tb_writer = None
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # Set up data-serving pipeline
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    eval_sampler = SequentialSampler(dev_dataset) if args.local_rank == -1 else DistributedSampler(dev_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    eval_dataloader = DataLoader(dev_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Estimate the number of update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Set-up weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Set-up optimizer and schedule (linear warmup)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_pct is None:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=math.floor(args.warmup_pct * t_total), num_training_steps=t_total)

    # Set-up mixed precision training
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Please install apex from https://www.github.com/nvidia/apex to use fp16 training.')
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Set-up multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Set-up distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info('***** Running training *****')
    logger.info('  Num examples = %d', len(train_dataset))
    logger.info('  Num Epochs = %d', args.num_train_epochs)
    logger.info('  Learning rate = {}'.format(args.learning_rate))
    logger.info('  Instantaneous batch size per GPU = %d', args.per_gpu_train_batch_size)
    logger.info('  Total train batch size (w. parallel, distributed & accumulation) = %d',
                args.train_batch_size * args.gradient_accumulation_steps *
                (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info('  Gradient Accumulation steps = %d', args.gradient_accumulation_steps)
    logger.info('  Total optimization steps = %d', t_total)

    # Initialize tracking variables
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    curr_eval_loss = 0.0
    best_eval_loss = float('inf')
    stale_epochs = 0

    # Iterate through training data
    best_checkpoint_path = None
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch', disable=args.local_rank not in [-1, 0])
    for _ in train_iterator:
        epoch_iterator = \
            tqdm(train_dataloader, desc='Iteration', disable=args.local_rank not in [-1, 0], mininterval=10, ncols=100)
        for step, batch in enumerate(epoch_iterator):
            # Perform a single training step
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if model_type != 'gpt2':
                # Prepare decoder inputs and labels for enc-dec models
                inputs['labels'] = batch[3][:, 1:].contiguous()  # shift
                decoder_input_ids = batch[3][:, :-1].clone()  # shift
                decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id  # remove masking
                inputs['decoder_input_ids'] = decoder_input_ids.contiguous()

            outputs = model(**inputs)
            loss = outputs[0]

            # Compute distributed loss
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Compute mixed-precision loss
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Back-propagate
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if global_step > args.max_steps > 0:
                epoch_iterator.close()
                break

        # Log metrics and evaluate model performance on the validation set
        mean_epoch_loss = (tr_loss - logging_loss) / len(epoch_iterator)
        logging.info('\n' + '*' * 10)
        logging.info('Mean epoch training loss: {:.4f}'.format(mean_epoch_loss))
        logging.info('*' * 10 + '\n')

        # Log metrics and evaluate model performance on the validation set
        if args.local_rank in [-1, 0]:
            # Only evaluate when single GPU otherwise metrics may not average well
            if args.local_rank == -1 and args.evaluate_during_training:
                results, curr_eval_loss, _, _ = evaluate(args, model, tokenizer, dev_dataset, eval_dataloader,
                                                      task_name, model_type, 'dev', global_step)
                for key, value in results.items():
                    tb_writer.add_scalar('eval_{}'.format(key), value[-1], global_step)
            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
            tb_writer.add_scalar('train_loss', mean_epoch_loss, global_step)
            tb_writer.add_scalar('eval_loss', curr_eval_loss, global_step)
            logging_loss = tr_loss

        # Save model checkpoint
        if args.local_rank in [-1, 0]:
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(task_name, global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # Take care of distributed / parallel training
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args_{}.bin'.format(task_name)))
            logger.info('Saving model checkpoint to %s', output_dir)
            # Delete old checkpoints to maintain a low memory footprint
            _rotate_checkpoints(args, 'checkpoint-{}'.format(task_name), use_mtime=False)

        # Check whether to stop training early
        if curr_eval_loss < best_eval_loss:
            best_eval_loss = curr_eval_loss
            stale_epochs = 0
            best_checkpoint_path = \
                os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(task_name, global_step))
        else:
            stale_epochs += 1
            logging.info(
                '!!! Development loss has not improved this epoch. Stale epochs: {:d} !!!'.format(stale_epochs))

        if stale_epochs >= args.patience:
            logging.info(
                '\n***** STOPPING TRAINING EARLY AFTER {:d} STALE VALIDATION STEPS *****\n'.format(stale_epochs))
            break

        if global_step > args.max_steps > 0:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_checkpoint_path


def evaluate(args, model, tokenizer, eval_dataset, eval_dataloader, task_name, model_type, split, step):
    """ Evaluates models on dev / test sets. """
    model.eval()
    processor = MoralStoriesProcessor()
    results = dict()
    softmax = torch.nn.Softmax(dim=1)

    # Eval!
    logger.info('***** Running evaluation on the validation / test set *****')
    logger.info('  Num examples = %d', len(eval_dataset))
    logger.info('  Batch size = %d', args.eval_batch_size)
    batch_losses = list()
    eval_loss = 0.0
    micro_loss, macro_loss = 0.0, 0.0
    num_batches, num_tokens = 0, 0
    preds = None
    soft_preds = None
    out_label_ids = None
    # Perform a single evaluation step
    for batch in tqdm(eval_dataloader, desc='Evaluating', mininterval=10, ncols=100):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if 'gen' not in task_name:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if model_type == 'bert' else None,
                          'labels': batch[3]}
            else:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if 'gpt2' not in model_type:
                    # Prepare decoder inputs and labels for enc-dec models
                    inputs['labels'] = batch[3][:, 1:].contiguous()  # shift
                    decoder_input_ids = batch[3][:, :-1].clone()  # shift
                    decoder_input_ids[decoder_input_ids == -100] = tokenizer.pad_token_id  # remove masking
                    inputs['decoder_input_ids'] = decoder_input_ids.contiguous()

            outputs = model(**inputs)

        tmp_eval_loss, logits = outputs[:2]
        soft_logits = softmax(logits)
        eval_loss += tmp_eval_loss.mean().item()
        batch_losses.append(tmp_eval_loss.item())

        if 'gen' not in task_name:
            if preds is None:
                preds = logits.detach().cpu().numpy()
                soft_preds = soft_logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                soft_preds = np.append(soft_preds, soft_logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        else:
            # Obtain per-token loss for perplexity computation
            batch_loss = get_token_loss(args, logits, batch[3], batch[4], model_type=model_type)
            macro_loss += batch_loss.mean().item()
            micro_loss += batch_loss.sum().item()
            num_batches += 1
            num_tokens += batch_loss.view(-1).shape[0]

    # Compute and update evaluation metric values
    if 'gen' not in task_name:
        # Isolate model predictions
        preds = np.argmax(preds, axis=1)
        soft_preds = soft_preds.tolist()
        curr_result = compute_cls_metrics(preds, out_label_ids)
    else:
        macro_perplexity = torch.exp(torch.tensor(macro_loss / num_batches)).item()
        micro_perplexity = torch.exp(torch.tensor(micro_loss / num_tokens)).item()
        curr_result = {'macro_perplexity': macro_perplexity,
                       'micro_perplexity': micro_perplexity}

    if len(results.keys()) == 0:
        for k, v in curr_result.items():
            results[k] = [v]
    else:
        for k, v in curr_result.items():
            results[k].append(v)

    # Log metrics
    output_eval_file = os.path.join(args.output_dir, 'results_{}_{}.txt'.format(task_name, split))
    with open(output_eval_file, 'a') as writer:
        logger.info('***** Eval results *****')
        writer.write('STEP: {:s}\n'.format(str(step)))
        for key in sorted(curr_result.keys()):
            logger.info('  %s = %s', key, str(curr_result[key]))
            writer.write('%s = %s\n' % (key, str(curr_result[key])))

    # Log predictions
    if 'gen' not in task_name:
        output_pred_file = \
            os.path.join(args.output_dir, 'predictions_{}_{}_{}.lst'.format(task_name, split, step))
        with open(output_pred_file, 'w') as writer:
            logger.info('***** Write predictions *****')
            for pred in preds:
                writer.write('{}\n'.format(processor.get_labels()[pred]))

    # Maintain a single metrics file
    if os.path.exists(args.output_dir):
        with open(os.path.join(args.output_dir, 'metrics_{}_{}.json'.format(task_name, split)), 'w') as f:
            f.write(json.dumps(results))
        f.close()

    # Report mean dev loss
    mean_eval_loss = eval_loss / len(eval_dataloader)
    logging.info('\n' + '*' * 10)
    logging.info('Mean development loss: {:.4f}'.format(mean_eval_loss))
    logging.info('*' * 10 + '\n')

    return results, mean_eval_loss, preds, soft_preds


def test_gen(args, model, tokenizer, dataset, dataloader, task_name, model_type, split, step, gen_per_prefix=1):
    """ Evaluates generative models. """

    # Test!
    logger.info('***** Testing generation on the test set *****')
    logger.info('  Num examples = %d', len(dataset))
    logger.info('  Batch size = %d', args.eval_batch_size)
    generation_inputs = list()
    generation_targets = list()
    generated_sequences = list()

    # Iterate through the test corpus
    model.eval()
    for batch_id, batch in enumerate(tqdm(dataloader, desc='Testing', mininterval=10, ncols=100)):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3],
                      'gen_prompt': batch[5]}

            # Modify inputs (assumes a batch size of 1)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            if model_type == 'gpt2' or 'action|' in task_name:
                pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
                if pad_token_id is None:
                    pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]
                try:
                    input_ids = inputs['input_ids'].tolist()
                    first_pad_idx = input_ids[0].index(pad_token_id)
                    input_ids = torch.tensor([input_ids[0][: first_pad_idx]], dtype=torch.long).to(args.device)
                    attention_mask = inputs['attention_mask'].tolist()
                    attention_mask = \
                        torch.tensor([attention_mask[0][: first_pad_idx]], dtype=torch.long).to(args.device)
                except ValueError:
                    input_ids = inputs['input_ids']
                    attention_mask = inputs['attention_mask']

            max_gen_length = args.max_gen_length
            if model_type == 'gpt2':
                max_gen_length += torch.max(torch.sum(attention_mask, axis=-1)).item()
            batch_generations = list()
            for _ in range(gen_per_prefix):
                if model_type == 'gpt2':
                    outputs = model.generate(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             min_length=5,
                                             max_length=max_gen_length,
                                             temperature=args.temperature,
                                             top_k=args.k if args.k > 0 else None,
                                             top_p=args.p if args.p > 0 else None,
                                             num_beams=args.num_beams if args.num_beams > 0 else None,
                                             do_sample=args.do_sample,
                                             early_stopping=True,
                                             no_repeat_ngram_size=3)
                else:
                    gen_prompt = \
                        inputs['gen_prompt'].item() if 'action|' in task_name else inputs['gen_prompt'][0].item()
                    outputs = model.generate(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             min_length=5,
                                             max_length=max_gen_length,
                                             temperature=args.temperature,
                                             top_k=args.k if args.k > 0 else None,
                                             top_p=args.p if args.p > 0 else None,
                                             num_beams=args.num_beams if args.num_beams > 0 else None,
                                             do_sample=args.do_sample,
                                             early_stopping=True,
                                             no_repeat_ngram_size=3,
                                             decoder_start_token_id=gen_prompt)

                # Remove the batch dimension when returning multiple sequences
                if len(outputs.shape) > 2:
                    outputs.squeeze_()
                batch_generations.append(outputs)

        # Convert model predictions to text sequences
        input_ids = inputs['input_ids'].tolist()
        target_ids = inputs['labels'].tolist()
        # Post-process model predictions and prediction targets
        batch_predictions = list()
        len_gen_input = 0
        for pass_id, pass_output in enumerate(batch_generations):
            pass_predictions = list()
            for generated_sequence_idx, generated_sequence in enumerate(pass_output):
                generated_sequence = generated_sequence.tolist()
                # GPT2
                if model_type == 'gpt2':
                    if pass_id == 0:
                        # Prepare inputs
                        gen_input = input_ids[generated_sequence_idx]
                        try:
                            gen_input = gen_input[: gen_input.index(tokenizer.eos_token_id)]
                        except ValueError:
                            pass
                        generation_inputs.append(tokenizer.decode(gen_input, clean_up_tokenization_spaces=True))
                        len_gen_input = len(gen_input)

                    # Prepare predictions
                    generated_sequence = generated_sequence[len_gen_input:]
                    try:
                        if generated_sequence.index(tokenizer.eos_token_id) == 0:
                            generated_sequence = generated_sequence[1:]
                        generated_sequence = generated_sequence[: generated_sequence.index(tokenizer.eos_token_id)]
                    except ValueError:
                        pass
                    pass_predictions.append(tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True))

                    if pass_id == 0:
                        # Prepare generation targets
                        gen_target = target_ids[generated_sequence_idx][len_gen_input:]
                        try:
                            gen_target = gen_target[: gen_target.index(tokenizer.eos_token_id)]
                        except ValueError:
                            pass
                        generation_targets.append(tokenizer.decode(gen_target, clean_up_tokenization_spaces=True))

                # For T5, split-off the initial <pad> token
                if model_type in ['t5', 'bart']:
                    # Prepare predictions
                    try:
                        generated_sequence = generated_sequence[: generated_sequence.index(tokenizer.eos_token_id)]
                    except ValueError:
                        pass
                    pass_predictions.append(
                        tokenizer.decode(generated_sequence[1:], clean_up_tokenization_spaces=True))

                    if pass_id == 0:
                        # Prepare inputs
                        gen_input = input_ids[generated_sequence_idx]
                        try:
                            if model_type == 't5':
                                gen_input = gen_input[: gen_input.index(tokenizer.eos_token_id)]
                            else:
                                gen_input = gen_input[: gen_input.index(tokenizer.pad_token_id)]
                        except ValueError:
                            pass
                        generation_inputs.append(tokenizer.decode(gen_input, clean_up_tokenization_spaces=True))

                        # Prepare generation targets
                        gen_target = target_ids[generated_sequence_idx]
                        try:
                            gen_target = gen_target[: gen_target.index(tokenizer.eos_token_id)]
                        except ValueError:
                            pass
                        generation_targets.append(tokenizer.decode(gen_target[1:], clean_up_tokenization_spaces=True))

            batch_predictions.append(pass_predictions)
        generated_sequences.append(batch_predictions)

    # Report sample generation results
    first_pass_predictions = list()
    for bp in generated_sequences:
        first_pass_predictions += bp[0]
    logging.info('***** Example generations (first pass only) *****')
    for s_id, gen_input in enumerate(generation_inputs):
        if s_id >= 10:
            break
        logging.info('  Inputs: {:s}'.format(gen_input))
        logging.info('  Reference: {:s}'.format(generation_targets[s_id]))
        logging.info('  Prediction: {:s}'.format(first_pass_predictions[s_id]))

    # Compute and update evaluation metric values
    curr_result = compute_gen_metrics(first_pass_predictions, generation_targets)

    # Log metrics
    output_eval_file = \
        os.path.join(args.output_dir, 'generation_test_results_{}_{}_{}.txt'.format(task_name, split, step))
    with open(output_eval_file, 'w') as writer:
        logger.info('***** Test results (first pass only) *****')
        writer.write('STEP: {:s}\n'.format(str(step)))
        for key in sorted(curr_result.keys()):
            logger.info('  %s = %s', key, str(curr_result[key]))
            writer.write('%s = %s\n' % (key, str(curr_result[key])))

    # Log predictions
    output_pred_file = \
        os.path.join(args.output_dir, 'generation_test_predictions_{}_{}_{}.lst'.format(task_name, split, step))
    with open(output_pred_file, 'w') as writer:
        logger.info('***** Write predictions (first pass only) *****')
        for gsi, gs in enumerate(first_pass_predictions):
            writer.write(json.dumps({'prefix': generation_inputs[gsi],
                                     'target': generation_targets[gsi],
                                     'prediction': gs}) + '\n')

    # For simplicity
    if gen_per_prefix == 1:
        generated_sequences = first_pass_predictions

    return generated_sequences, generation_inputs, generation_targets


def consequence_gen_with_cls_refinement(args, eval_split):
    """ Refines action predictions through a multi-stage generation process. """

    # Generate consequence based on story prefix
    args.consequence_generator_model_type = args.consequence_generator_model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.consequence_generator_model_type]
    global_step = int(args.consequence_generator_checkpoint.split('-')[-1])
    model = model_class.from_pretrained(args.consequence_generator_checkpoint)
    tokenizer = tokenizer_class.from_pretrained(args.consequence_generator_checkpoint)
    model.to(args.device)

    specified_batch_size = args.per_gpu_eval_batch_size

    # Set up data-serving pipeline
    split_consequence_predictions = {'train': None, 'dev': None, 'test': None}
    for split in ['train', 'dev', 'test']:
        dataset = load_and_cache_examples(args, tokenizer, split, 'consequence|action+context_gen',
                                          args.consequence_generator_model_type)
        args.per_gpu_eval_batch_size = 1  # set batch size to 1
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)

        # Generate consequence predictions
        logging.info('\n' + '*' * 20)
        logging.info('Generating initial consequence predictions using the model from checkpoint {} for split {}'
                     .format(args.consequence_generator_checkpoint, split))
        logging.info('*' * 20 + '\n')

        split_consequence_predictions[split] = test_gen(args,
                                                        model,
                                                        tokenizer,
                                                        dataset,
                                                        dataloader,
                                                        'consequence|action+context_gen',
                                                        args.consequence_generator_model_type,
                                                        split,
                                                        global_step)[0]

    # Include model generations into the input data for the consequence classifier
    args.consequence_classifier_model_type = args.consequence_classifier_model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.consequence_classifier_model_type]
    model = model_class.from_pretrained(args.consequence_classifier_checkpoint)
    tokenizer = tokenizer_class.from_pretrained('roberta-large')
    model.to(args.device)

    # Incorporate consequence predictions into training and dev data
    consequence_classification_datasets = {'train': None, 'dev': None, 'test': None}
    consequence_classification_dataloaders = {'train': None, 'dev': None, 'test': None}
    split_consequence_labels = {'train': None, 'dev': None, 'test': None}
    args.per_gpu_eval_batch_size = specified_batch_size
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    for split in ['train', 'dev', 'test']:
        dataset = load_and_cache_examples(args, tokenizer, split, 'consequence+action+context_cls',
                                          args.consequence_classifier_model_type,
                                          predictions=['consequences', split_consequence_predictions[split]])
        consequence_classification_datasets[split] = dataset
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)
        consequence_classification_dataloaders[split] = dataloader

        # Generate consequence predictions
        logging.info('\n' + '*' * 20)
        logging.info('Generating consequence labels using the model from checkpoint {} for split {}'
                     .format(args.consequence_classifier_checkpoint, split))
        logging.info('*' * 20 + '\n')

        split_consequence_labels[split] = evaluate(args,
                                                   model,
                                                   tokenizer,
                                                   dataset,
                                                   dataloader,
                                                   'consequence+action+context_cls',
                                                   args.consequence_classifier_model_type,
                                                   split,
                                                   dataset)[2]

    # Include model predictions into the input data
    args.consequence_refiner_model_type = args.consequence_refiner_model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.consequence_refiner_model_type]
    config = config_class.from_pretrained(args.consequence_refiner_model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.consequence_refiner_model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.consequence_refiner_model_name_or_path,
                                        from_tf=bool('.ckpt' in args.consequence_refiner_model_name_or_path),
                                        config=config)
    new_tokens = ['<|NRM|>', '<|SIT|>', '<|ITN|>',
                  '<|M_ACT|>', '<|I_ACT|>',
                  '<|M_CSQ|>', '<|I_CSQ|>',
                  '<|ACT|>', '<|CSQ|>',
                  '<|CSQ_TRUE|>', '<|CSQ_FALSE|>']
    tokens_to_add = list()
    tokenizer_vocab = tokenizer.get_vocab()
    # Check if tokens exist in tokenizer vocab
    for tok in new_tokens:
        if tokenizer_vocab.get(tok, None) is None:
            tokens_to_add.append(tok)
    # Add to tokenizer vocab
    tokenizer.add_special_tokens({'additional_special_tokens': tokens_to_add})
    # Initialize new embeddings
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    # Incorporate consequence predictions into training and dev data
    consequence_refinement_datasets = {'train': None, 'dev': None, 'test': None}
    consequence_refinement_dataloaders = {'train': None, 'dev': None, 'test': None}#
    args.per_gpu_eval_batch_size = 1  # set batch size to 1
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    for split in ['train', 'dev', 'test']:
        dataset = load_and_cache_examples(args, tokenizer, split, 'consequence|action+context_genref',
                                          args.consequence_refiner_model_type,
                                          predictions=[('consequences', split_consequence_predictions[split]),
                                                       ('consequence_labels', split_consequence_labels[split])])
        consequence_refinement_datasets[split] = dataset
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)
        consequence_refinement_dataloaders[split] = dataloader

    # Train the consequence refinement model
    logging.info('\n' + '*' * 20)
    logging.info('Training a model to generate refined consequence predictions based on the draft consequence and label')
    logging.info('*' * 20 + '\n')
    global_step, tr_loss, act_best_checkpoint_name = train(args, model, tokenizer, 'consequence|action+context_genref',
                                                           args.consequence_refiner_model_type,
                                                           consequence_refinement_datasets['train'],
                                                           consequence_refinement_datasets['dev'])
    logger.info(' global_step = %s, average loss = %s', global_step, tr_loss)

    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(act_best_checkpoint_name)
    tokenizer = tokenizer_class.from_pretrained(act_best_checkpoint_name)
    model.to(args.device)

    # 5. Generate refined consequences
    logging.info('\n' + '*' * 20)
    logging.info('Generating refined consequences using the model from checkpoint {}'
                 .format(act_best_checkpoint_name))
    logging.info('*' * 20 + '\n')
    refined_consequences, generation_inputs, generation_targets = \
        test_gen(args,
                 model,
                 tokenizer,
                 consequence_refinement_datasets[eval_split],
                 consequence_refinement_dataloaders[eval_split],
                 'consequence|action+context_genref',
                 args.consequence_refiner_model_type,
                 split,
                 global_step,
                 gen_per_prefix=args.num_consequences)

    # Sort consequence predictions
    consequence_pass_generations = {pass_id: list() for pass_id in range(args.num_consequences)}
    for batch_generations in refined_consequences:
        for pi in range(args.num_consequences):
            consequence_pass_generations[pi] += batch_generations[pi]

    # Rank predicted consequences using a pre-trained classifier
    logging.info('\n' + '*' * 20)
    logging.info('Ranking consequence predictions using a pre-trained classifier')
    logging.info('*' * 20 + '\n')
    # Load classifier
    args.consequence_classifier_model_type = args.consequence_classifier_model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.consequence_classifier_model_type]
    global_step = int(args.consequence_classifier_checkpoint.split('-')[-1])
    model = model_class.from_pretrained(args.consequence_classifier_checkpoint)
    try:
        tokenizer = tokenizer_class.from_pretrained(args.consequence_classifier_checkpoint)
    except Exception:
        tokenizer = tokenizer_class.from_pretrained('roberta-large')  # hack
    model.to(args.device)

    consequence_score_table = {ex_id: list() for ex_id in range(len(consequence_pass_generations[0]))}
    for pass_id in range(args.num_consequences):
        initial_consequence_predictions = consequence_pass_generations[pass_id]
        # Set up data-serving pipeline
        eval_dataset = load_and_cache_examples(args, tokenizer, split, 'consequence+action+context_cls',
                                               args.consequence_generator_model_type,
                                               predictions=['consequences', initial_consequence_predictions])
        eval_sampler = SequentialSampler(eval_dataset)
        args.per_gpu_eval_batch_size = specified_batch_size
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Obtain clasisfier predictions
        results, mean_eval_loss, scores, softmax_scores = \
            evaluate(args, model, tokenizer, eval_dataset, eval_dataloader, 'consequence+action+context_cls',
                     args.consequence_classifier_model_type, split, global_step)

        # Assign scores to consequences
        for csq_id, consequence in enumerate(initial_consequence_predictions):
            consequence_score_table[csq_id].append((consequence, None, softmax_scores[csq_id][1]))

    # 4. Return the consequence corresponding to the 'most good' consequence as generation output
    logging.info('\n' + '*' * 20)
    logging.info('Picking the best consequences from the predicted alternatives')
    logging.info('*' * 20 + '\n')
    # Log predictions
    best_predictions = list()
    output_pred_file = \
        os.path.join(args.output_dir, 'best_ranked_consequences_{}.lst'.format(split))
    with open(output_pred_file, 'w') as writer:
        logger.info('***** Write predictions *****')
        for story_id, ac_list in consequence_score_table.items():
            # Sort consequence predictions
            sorted_consequences = sorted(ac_list, reverse=True, key=lambda x: x[2])
            best_predictions.append(sorted_consequences[0][0])
            # Write to file
            writer.write(json.dumps({'prefix': generation_inputs[story_id],
                                     'target': generation_targets[story_id],
                                     'prediction': sorted_consequences[0][0]}) + '\n')
            if story_id < 10:
                logging.info('***** Example ranking *****')
                logging.info('Story prefix: {:s}'.format(generation_inputs[story_id]))
                logging.info('Gold reference consequence: {:s}'.format(generation_targets[story_id]))
                logging.info('Action predictions by the generator, '
                             'ranked by the \'goodness\' of their anticipated consequences:')
                for tpl_id, tpl in enumerate(sorted_consequences):
                    logging.info('-' * 10)
                    logging.info('Rank {:d}'.format(tpl_id))
                    logging.info('Predicted consequence: {:s}'.format(tpl[0]))
                    logging.info('Anticipated consequence: {:s}'.format(str(tpl[1])))
                    logging.info('Score: {:.4f}'.format(tpl[2]))

    # Compute and update evaluation metric values
    best_result = compute_gen_metrics(best_predictions, generation_targets)
    # Log metrics
    output_eval_file = os.path.join(args.output_dir, 'generation_test_results_{}_{}_{}.txt'.format(
        'consequence_refinement', split, global_step))
    with open(output_eval_file, 'w') as writer:
        logger.info('***** Test results (best consequences) *****')
        writer.write('STEP: {:s}\n'.format(str(global_step)))
        for key in sorted(best_result.keys()):
            logger.info('  %s = %s', key, str(best_result[key]))
            writer.write('%s = %s\n' % (key, str(best_result[key])))

    return split_consequence_predictions['test'], best_predictions


def load_and_cache_examples(args, tokenizer, split, task_name, model_type, predictions=None):
    """ Prepares the dataset splits for use with the model. """
    processor = MoralStoriesProcessor()
    if task_name != 'consequence|action+context_genref':
        args.data_dir = os.path.join(args.original_data_dir, task_name, args.split_name)
    else:
        args.data_dir = os.path.join(args.original_data_dir, 'consequence|action+context_gen', args.split_name)

    # Get features
    logger.info('Creating features from dataset file at %s', args.data_dir)
    label_list = processor.get_labels()
    if split == 'train':
        examples = processor.get_train_examples(args.data_dir)
    elif split == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    elif split == 'test':
        examples = processor.get_test_examples(args.data_dir)
    else:
        raise Exception('split value should be in [train, dev, test]')

    # Replace gold sequences with model predictions
    if predictions is not None:
        if type(predictions[0]) != tuple:
            all_predictions = [tuple(predictions)]
        else:
            all_predictions = predictions
        extended_examples = list()

        for predictions in all_predictions:
            if predictions[0] == 'consequences':
                if len(all_predictions) == 1:
                    # Remove negative examples
                    positive_examples = list()
                    for ex in examples:
                        if ex.label == '1':
                            positive_examples.append(ex)
                    examples = positive_examples

                for pr_id, pr in enumerate(predictions[1]):
                    ex = examples[pr_id]
                    if ex.moral_consequence is not None:
                        if len(all_predictions) == 1:
                            ex.moral_consequence = pr
                        else:
                            ex.moral_consequence_draft = pr
                    else:
                        if len(all_predictions) == 1:
                            ex.immoral_consequence = pr
                        else:
                            ex.immoral_consequence_draft = pr
                    extended_examples.append(ex)
                examples = extended_examples
                extended_examples = list()

            if predictions[0] == 'consequence_labels':
                for pr_id, pr in enumerate(predictions[1]):
                    ex = examples[pr_id]
                    if ex.moral_consequence_draft is not None:
                        if pr == 1:
                            ex.moral_consequence_draft = ex.moral_consequence_draft + ' ' + '<|CSQ_TRUE|>'
                        else:
                            ex.moral_consequence_draft = ex.moral_consequence_draft + ' ' + '<|CSQ_FALSE|>'
                    else:
                        if pr == 0:
                            ex.immoral_consequence_draft = ex.immoral_consequence_draft + ' ' + '<|CSQ_TRUE|>'
                        else:
                            ex.immoral_consequence_draft = ex.immoral_consequence_draft + ' ' + '<|CSQ_FALSE|>'
                    extended_examples.append(ex)
                examples = extended_examples
                extended_examples = list()

    # Generate features; target task is classification
    pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
    if pad_token_id is None:
        pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.eos_token])[0]
    features = convert_examples_to_features(examples,
                                            label_list,
                                            args.max_seq_length,
                                            args.max_gen_length,
                                            tokenizer,
                                            task_name,
                                            model_type,
                                            TASK_DICT[task_name],
                                            cls_token_at_end=False,
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta']),
                                            cls_token_segment_id=0,
                                            pad_on_left=False,
                                            pad_token=pad_token_id,
                                            pad_token_segment_id=0,
                                            is_eval=split == 'test',
                                            fit_to_max_corpus_len=True)

    # Make feature tensors
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    if 'gen' in task_name:
        all_label_masks = torch.tensor([f.label_mask for f in features], dtype=torch.long)
        all_gen_prompts = torch.tensor([f.gen_prompt_id for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_label_masks, all_gen_prompts)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument('--data_dir', default=None, type=str, required=True,
                        help='The input data dir. Should contain the .tsv files (or other data files) for the task.')
    parser.add_argument('--consequence_generator_model_type', default=None, type=str, required=True,
                        choices=list(MODEL_CLASSES.keys()),
                        help='Model type to use for initial consequence prediction, selected in the list: ' +
                             ', '.join(MODEL_CLASSES.keys()))
    parser.add_argument('--consequence_classifier_model_type', default=None, type=str, required=True,
                        choices=list(MODEL_CLASSES.keys()),
                        help='Model type to use for consequence classification, selected in the list: ' +
                             ', '.join(MODEL_CLASSES.keys()))
    parser.add_argument('--consequence_refiner_model_type', default=None, type=str, required=True,
                        choices=list(MODEL_CLASSES.keys()),
                        help='Model type to use for consequence refinement, selected in the list: ' +
                             ', '.join(MODEL_CLASSES.keys()))
    parser.add_argument('--consequence_generator_checkpoint', default=None, type=str, required=True,
                        help='Path to pre-trained model used for initial consequence generation')
    parser.add_argument('--consequence_classifier_checkpoint', default=None, type=str, required=True,
                        help='Path to pre-trained model used for initial consequence classification')
    parser.add_argument('--consequence_refiner_model_name_or_path', default=None, type=str, required=True,
                        help='Name of / path to pre-trained model to be fine-tuned')
    parser.add_argument('--split_name', default=None, type=str, required=True, choices=SPLITS,
                        help='The name of the data split used to train / evaluate the model: ' + ', '.join(SPLITS))
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='The root output directory where the model predictions and checkpoints will be written.')

    ## Generation parameters
    parser.add_argument('--max_gen_length', default=60, type=int,
                        help='The maximum length of the sequence to be generated.')
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='The value used to module the next token probabilities.')
    parser.add_argument('--k', default=0, type=int,
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
    parser.add_argument('--p', default=0, type=float,
                        help='If set to float < 1, only the most probable tokens with probabilities that add up to '
                             'top_p or higher are kept for generation.')
    parser.add_argument('--num_beams', default=0, type=int, required=False, help='beams for beam search')
    parser.add_argument('--do_sample', action='store_true',
                        help='Whether to generate predictions via sampling; if off, decoding is done greedily.')
    parser.add_argument('--patience', default=-1, type=int, required=False,
                        help='Number of epochs without dev-set loss improvement to ignore before '
                             'stopping the training.')
    parser.add_argument('--num_consequences', default=0, type=int, required=False,
                        help='number of consequences to ge generated for a single story prefix prior to ranking')

    ## Other parameters
    parser.add_argument('--config_name', default='', type=str,
                        help='Pretrained config name or path if not the same as model_name')
    parser.add_argument('--tokenizer_name', default='', type=str,
                        help='Pretrained tokenizer name or path if not the same as model_name')
    parser.add_argument('--cache_dir', default='', type=str,
                        help='The cache directory where do you want to store the pre-trained models downloaded from s3')
    parser.add_argument('--max_seq_length', default=128, type=int,
                        help='The maximum total input sequence length after tokenization. Sequences longer '
                             'than this will be truncated, sequences shorter will be padded.')
    parser.add_argument('--do_train', action='store_true',
                        help='Whether to run training.')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to run eval on the dev set.')
    parser.add_argument('--do_prediction', action='store_true',
                        help='Whether to run prediction on the test set. (Training will not be executed.)')
    parser.add_argument('--evaluate_during_training', action='store_true',
                        help='Do evaluation during training at each logging step.')
    parser.add_argument('--do_lower_case', action='store_true',
                        help='Set this flag if you are using an uncased model.')
    parser.add_argument('--run_on_test', action='store_true',
                        help='Evaluate model on the test split.')
    parser.add_argument('--data_cache_dir', default=None, type=str,
                        help='The root directory for caching features.')
    parser.add_argument('--pretrained_dir', default=None, type=str,
                        help='The directory containing the checkpoint of a pretrained model to be fine-tuned.')
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Maximum number of checkpoints to keep')

    parser.add_argument('--per_gpu_train_batch_size', default=8, type=int,
                        help='Batch size per GPU/CPU for training.')
    parser.add_argument('--per_gpu_eval_batch_size', default=8, type=int,
                        help='Batch size per GPU/CPU for evaluation.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='The initial learning rate for Adam.')
    parser.add_argument('--weight_decay', default=0.0, type=float,
                        help='Weight deay if we apply some.')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float,
                        help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='Max gradient norm.')
    parser.add_argument('--num_train_epochs', default=3.0, type=float,
                        help='Total number of training epochs to perform.')
    parser.add_argument('--max_steps', default=-1, type=int,
                        help='If > 0: set total number of training steps to perform. Override num_train_epochs.')
    parser.add_argument('--warmup_steps', default=0, type=int,
                        help='Linear warmup over warmup_steps.')
    parser.add_argument('--warmup_pct', default=None, type=float,
                        help='Linear warmup over warmup_pct * total_steps.')

    parser.add_argument('--logging_steps', type=int, default=50,
                        help='Log every X updates steps.')
    parser.add_argument('--save_steps', type=int, default=50,
                        help='Save checkpoint every X updates steps.')
    parser.add_argument('--eval_all_checkpoints', action='store_true',
                        help='Evaluate all checkpoints starting with the same prefix as model_name ending and ending '
                             'with step number')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Avoid using CUDA when available')
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help='Overwrite the content of the output directory')
    parser.add_argument('--overwrite_cache', action='store_true',
                        help='Overwrite the cached training and evaluation sets')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for initialization')

    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help='For fp16: Apex AMP optimization level selected in [\'O0\', \'O1\', \'O2\', and \'O3\'].'
                             'See details at https://nvidia.github.io/apex/amp.html')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='For distributed training: local_rank')
    parser.add_argument('--server_ip', type=str, default='', help='For distant debugging.')
    parser.add_argument('--server_port', type=str, default='', help='For distant debugging.')
    args = parser.parse_args()

    # Check if directories need to be created
    args.original_data_dir = args.data_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup distant debugging, if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        logging.info('Waiting for debugger attach ...')
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning('Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s',
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Generate with refinement
    logger.info('Generating consequences through iterative refinement:')
    initial_consequences, refined_consequences = consequence_gen_with_cls_refinement(args, 'test')
    logger.info('Self-BLEU between initial and refined consequence predictions:')
    logging.info(compute_bleu(initial_consequences, refined_consequences))
    logger.info('***** Experiment finished *****')


if __name__ == '__main__':
    main()
