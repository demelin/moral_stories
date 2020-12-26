from __future__ import absolute_import, division, print_function

import os
import json
import torch
import logging
import argparse

import numpy as np

from tqdm import tqdm
from torch.utils.data import (DataLoader,
                              SequentialSampler,
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
                          GPT2Tokenizer)

from utils import (compute_cls_metrics, compute_gen_metrics, convert_examples_to_features,
                   MoralStoriesProcessor, set_seed)


logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer),
    't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
}

TASKS = ['action|context_gen', 'action+context_cls']
TASK_DICT = {
                'action|context_gen': ['nrm', 'sit', 'itn', 'ma', 'ia'],
                'action+context_cls': ['nrm', 'sit', 'itn', 'ma', 'ia'],
}
SPLITS = ['norm_distance', 'lexical_bias', 'minimal_pairs']
MAX_GEN_LENGTH = int(10000)


def evaluate(args, model, eval_dataset, eval_dataloader, task_name, model_type, split, step):
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
    preds = None
    soft_preds = None
    out_label_ids = None
    # Perform a single evaluation step
    for batch in tqdm(eval_dataloader, desc='Evaluating', mininterval=10, ncols=100):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if model_type == 'bert' else None,
                      'labels': batch[3]}

            outputs = model(**inputs)

        tmp_eval_loss, logits = outputs[:2]
        soft_logits = softmax(logits)
        eval_loss += tmp_eval_loss.mean().item()
        batch_losses.append(tmp_eval_loss.item())

        if preds is None:
            preds = logits.detach().cpu().numpy()
            soft_preds = soft_logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            soft_preds = np.append(soft_preds, soft_logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    # Compute and update evaluation metric values
    # Isolate model predictions
    preds = np.argmax(preds, axis=1)
    soft_preds = soft_preds.tolist()
    curr_result = compute_cls_metrics(preds, out_label_ids)

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

    return results, mean_eval_loss, soft_preds


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
            if model_type == 'gpt2' or 'action' in task_name:
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


def action_gen_with_ranking(args, split):
    """ Generates moral / immoral actions by ranking a set of hypotheses. """

    # Use pre-trained action generation model with nucleus sampling and high-p (e.g. 0.95) to generate N (e.g. 10)
    # actions from the same story prefix
    # Instantiate model and tokenizer
    args.p = 0.90
    args.action_draft_model_type = args.action_draft_model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.action_draft_model_type]
    global_step = int(args.action_generator_checkpoint.split('-')[-1])
    model = model_class.from_pretrained(args.action_generator_checkpoint)
    tokenizer = tokenizer_class.from_pretrained(args.action_generator_checkpoint)
    model.to(args.device)

    specified_batch_size = args.per_gpu_eval_batch_size

    # Set up data-serving pipeline
    eval_dataset = load_and_cache_examples(args, tokenizer, split, 'action|context_gen', args.action_draft_model_type)
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    args.per_gpu_eval_batch_size = 1  # set batch size to 1
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Generate action predictions
    logging.info('\n' + '*' * 20)
    logging.info('Generating initial action predictions using the model from checkpoint {}'
                 .format(args.action_generator_checkpoint))
    logging.info('*' * 20 + '\n')

    action_predictions, generation_inputs, generation_targets = test_gen(args,
                                                                         model,
                                                                         tokenizer,
                                                                         eval_dataset,
                                                                         eval_dataloader,
                                                                         'action|context_gen',
                                                                         args.action_draft_model_type,
                                                                         split,
                                                                         global_step,
                                                                         gen_per_prefix=args.num_actions)

    # Sort action predictions
    action_pass_generations = {pass_id: list() for pass_id in range(args.num_actions)}
    for batch_generations in action_predictions:
        for pi in range(args.num_actions):
            action_pass_generations[pi] += batch_generations[pi]

    # Rank predicted actions using a pre-trained classifier
    logging.info('\n' + '*' * 20)
    logging.info('Ranking action predictions using a pre-trained classifier')
    logging.info('*' * 20 + '\n')
    # Load classifier
    args.action_classifier_model_type = args.action_classifier_model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.action_classifier_model_type]
    global_step = int(args.action_classifier_checkpoint.split('-')[-1])
    model = model_class.from_pretrained(args.action_classifier_checkpoint)
    try:
        tokenizer = tokenizer_class.from_pretrained(args.action_classifier_checkpoint)
    except Exception:
        tokenizer = tokenizer_class.from_pretrained('roberta-large')  # hack
    model.to(args.device)

    action_consequence_table = {ex_id: list() for ex_id in range(len(action_pass_generations[0]))}
    for pass_id in range(args.num_actions):
        initial_action_predictions = action_pass_generations[pass_id]
        # Set up data-serving pipeline
        eval_dataset = load_and_cache_examples(args, tokenizer, split, 'action+context_cls',
                                               args.consequence_generation_model_type,
                                               predictions=['actions', initial_action_predictions])
        eval_sampler = SequentialSampler(eval_dataset)
        args.per_gpu_eval_batch_size = specified_batch_size
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Obtain clasisfier predictions
        results, mean_eval_loss, softmax_scores = \
            evaluate(args, model, eval_dataset, eval_dataloader, 'action+context_cls',
                     args.action_classifier_model_type, split, global_step)
        # Assign scores to actions
        for act_id, action in enumerate(initial_action_predictions):
            score_id = 1 if act_id % 2 == 0 else 0
            action_consequence_table[act_id].append((action, None, softmax_scores[act_id][score_id]))

    # Return the action corresponding to the 'most good' consequence as generation output
    logging.info('\n' + '*' * 20)
    logging.info('Picking the best actions from the predicted alternatives')
    logging.info('*' * 20 + '\n')
    # Log predictions
    best_predictions = list()
    output_pred_file = \
        os.path.join(args.output_dir, 'best_ranked_actions_{}.lst'.format(split))
    with open(output_pred_file, 'w') as writer:
        logger.info('***** Write predictions *****')
        for story_id, ac_list in action_consequence_table.items():
            # Sort action predictions
            sorted_actions = sorted(ac_list, reverse=True, key=lambda x: x[2])
            best_predictions.append(sorted_actions[0][0])
            # Write to file
            writer.write(json.dumps({'prefix': generation_inputs[story_id],
                                     'target': generation_targets[story_id],
                                     'prediction': sorted_actions[0][0]}) + '\n')
            if story_id < 10:
                logging.info('***** Example ranking *****')
                logging.info('Story prefix: {:s}'.format(generation_inputs[story_id]))
                logging.info('Gold reference action: {:s}'.format(generation_targets[story_id]))
                logging.info('Ranked action predictions by the generator:')
                for tpl_id, tpl in enumerate(sorted_actions):
                    logging.info('-' * 10)
                    logging.info('Rank {:d}'.format(tpl_id))
                    logging.info('Predicted action: {:s}'.format(tpl[0]))
                    logging.info('Anticipated consequence: {:s}'.format(str(tpl[1])))
                    logging.info('Score: {:.4f}'.format(tpl[2]))

    # Compute and update evaluation metric values
    best_result = compute_gen_metrics(best_predictions, generation_targets)
    # Log metrics
    output_eval_file = os.path.join(args.output_dir, 'generation_test_results_{}_{}_{}.txt'.format(
        'action_refinement', split, global_step))
    with open(output_eval_file, 'w') as writer:
        logger.info('***** Test results (best actions) *****')
        writer.write('STEP: {:s}\n'.format(str(global_step)))
        for key in sorted(best_result.keys()):
            logger.info('  %s = %s', key, str(best_result[key]))
            writer.write('%s = %s\n' % (key, str(best_result[key])))


def load_and_cache_examples(args, tokenizer, split, task_name, model_type, predictions=None):
    """ Prepares the dataset splits for use with the model. """
    processor = MoralStoriesProcessor()

    args.data_dir = os.path.join(args.original_data_dir, task_name, args.split_name)

    # Get features
    logger.info('Creating features from dataset file at %s', args.data_dir)
    label_list = processor.get_labels()
    if split == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    elif split == 'test':
        examples = processor.get_test_examples(args.data_dir)
    else:
        raise Exception('split value should be in [dev, test]')

    # Replace gold sequences with model predictions
    if predictions is not None:
        if type(predictions[0]) != tuple:
            all_predictions = [tuple(predictions)]
        else:
            all_predictions = predictions
        extended_examples = list()
        for predictions in all_predictions:
            if predictions[0] == 'actions':
                for pr_id, pr in enumerate(predictions[1]):
                    ex = examples[pr_id]
                    if ex.label == '1':
                        ex.moral_action = pr
                    else:
                        ex.immoral_action = pr
                    extended_examples.append(ex)
            if predictions[0] == 'consequences':
                for pr_id, pr in enumerate(predictions[1]):
                    ex = examples[pr_id]
                    if ex.label == '1':
                        ex.moral_consequence = pr
                    else:
                        ex.immoral_consequence = pr
                    extended_examples.append(ex)
            examples = extended_examples

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
    parser.add_argument('--action_draft_model_type', default=None, type=str, required=True,
                        choices=list(MODEL_CLASSES.keys()),
                        help='Model type to use for initial action prediction, selected in the list: ' +
                             ', '.join(MODEL_CLASSES.keys()))
    parser.add_argument('--action_classifier_model_type', default=None, type=str, required=True,
                        choices=list(MODEL_CLASSES.keys()),
                        help='Model type to use for action classification, selected in the list: ' +
                             ', '.join(MODEL_CLASSES.keys()))
    parser.add_argument('--action_generator_checkpoint', default=None, type=str, required=True,
                        help='Path to pre-trained model used for initial action generation')
    parser.add_argument('--action_classifier_checkpoint', default=None, type=str, required=True,
                        help='Path to pre-trained model used for action classification')
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
    parser.add_argument('--sc101_action_embeddings_path', default=None, type=str,
                        help='Path to the file containing the Social-Chemistry-101 action embeddings.')
    parser.add_argument('--num_actions', default=0, type=int, required=False,
                        help='number of actions to ge generated for a single story prefix prior to ranking')
    parser.add_argument('--predict_consequences', action='store_true',
                        help='Whether to use consequences when ranking predicted action alternatives.')

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
    parser.add_argument('--do_lower_case', action='store_true',
                        help='Set this flag if you are using an uncased model.')
    parser.add_argument('--data_cache_dir', default=None, type=str,
                        help='The root directory for caching features.')

    parser.add_argument('--per_gpu_eval_batch_size', default=8, type=int,
                        help='Batch size per GPU/CPU for evaluation.')

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
    logger.info('Generating actions through consequence ranking:')
    action_gen_with_ranking(args, 'test')
    logger.info('***** Experiment finished *****')


if __name__ == '__main__':
    main()
