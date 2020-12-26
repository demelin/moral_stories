# coding=utf-8

from __future__ import absolute_import, division, print_function

import os
import re
import csv
import sys
import glob
import json
import torch
import random
import shutil
import logging
import sacrebleu

import numpy as np

from io import open
from sacrerouge.metrics import Rouge
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss

SPECIAL_GEN_TOKENS = {'nrm': '<|NRM|>',
                      'sit': '<|SIT|>',
                      'itn': '<|ITN|>',
                      'ma': '<|M_ACT|>',
                      'ia': '<|I_ACT|>',
                      'mc': '<|M_CSQ|>',
                      'ic': '<|I_CSQ|>',
                      'action': '<|ACT|>',
                      'consequence': '<|CSQ|>',
                      'mad': '<|M_ACT|>',
                      'iad': '<|I_ACT|>',
                      'mcd': '<|CSQ|>',
                      'icd': '<|CSQ|>'}

logger = logging.getLogger(__name__)


class InputExample(object):
    """ A single training/test example for simple sequence classification. """

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class SocialChemExample(object):
    """ Constructs a InputExample for SocialChemistry101 action classification. """

    def __init__(self, guid, action, label=None):
        self.guid = guid
        self.action = action
        self.label = label


class MoralStoryExample(object):
    """A single training/test example for classification of moral stories."""

    def __init__(self, guid,
                 norm, situation, intention,
                 moral_action, immoral_action,
                 moral_consequence, immoral_consequence,
                 label):
        """ Constructs a MoralStoryExample. Several of the inputs are optional depending on the modeled task.

        Args:
            guid: Unique id for the example.
            norm: string. The moral norm central to the story.
            situation: string. Story situation sentence.
            intention: string. Story intention sentence.
            moral_action: string. Action that follows the moral norm.
            immoral_action: string. Action that violates the moral norm.
            moral_consequence: string. Consequence of following the action.
            immoral_consequence: string. Consequence of violating the action.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.norm = norm
        self.situation = situation
        self.intention = intention

        self.moral_action = moral_action
        self.immoral_action = immoral_action

        self.moral_consequence = moral_consequence
        self.immoral_consequence = immoral_consequence

        self.label = label

        # Special
        self.moral_action_draft = None
        self.immoral_action_draft = None
        self.moral_consequence_draft = None
        self.immoral_consequence_draft = None


class InputFeatures(object):
    """ A single set of features of data. """

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, label_mask, gen_prompt_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_mask = label_mask
        self.gen_prompt_id = gen_prompt_id


class DataProcessor(object):
    """ Base class for data converters for sequence classification data sets. """

    def get_train_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the train set. """
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the dev set. """
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """ Gets a collection of `InputExample`s for the test set. """
        raise NotImplementedError()

    def get_labels(self):
        """ Gets the list of labels for this data set. """
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """ Reads a tab separated value file. """
        with open(input_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter='\t', quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
        return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        """ Reads a .jsonl file. """
        records = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        return records


class SocialChemProcessor(DataProcessor):
    """ Converts Social-Chemistry-101 actions for sequence classification tasks. """

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'train.jsonl')))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'dev.jsonl')))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'test.jsonl')))

    def get_labels(self):
        # 0 may correspond to 'bad', while 1 may correspond to 'good'
        return ['0', '1']

    @staticmethod
    def _create_examples(records):
        # Convert corpus contents to examples
        examples = list()
        for i, record in enumerate(records):
            guid = record['qID']
            action = record.get('action', None)
            label = record.get('label', None)

            if label is None:
                # This is a dummy label for test prediction.
                # test.jsonl doesn't include the `answer`.
                label = '0'

            examples.append(SocialChemExample(guid=guid,
                                              action=action,
                                              label=label))
        return examples


class MoralStoriesProcessor(DataProcessor):
    """ Converts moral stories for sequence classification tasks. """

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'train.jsonl')))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'dev.jsonl')))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, 'test.jsonl')))

    def get_labels(self):
        # 0 may correspond to 'immoral' / 'implausible', while 1 may correspond to 'moral' / 'plausible'
        return ['0', '1']

    def create_examples(self, records):
        return self._create_examples(records)

    @staticmethod
    def _create_examples(records):
        # Convert corpus contents to examples
        examples = list()
        for i, record in enumerate(records):
            guid = record['ID']
            norm = record.get('norm', None)
            situation = record.get('situation', None)
            intention = record.get('intention', None)

            moral_action = record.get('moral_action', None)
            immoral_action = record.get('immoral_action', None)

            moral_consequence = record.get('moral_consequence', None)
            immoral_consequence = record.get('immoral_consequence', None)

            label = record.get('label', None)

            if label is None:
                # This is a dummy label for test prediction.
                # test.jsonl doesn't include the `answer`.
                label = '0'

            examples.append(MoralStoryExample(guid=guid,
                                              norm=norm, situation=situation, intention=intention,
                                              moral_action=moral_action, immoral_action=immoral_action,
                                              moral_consequence=moral_consequence,
                                              immoral_consequence=immoral_consequence,
                                              label=label))
        return examples


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 max_tgt_length,
                                 tokenizer,
                                 task_name,
                                 model_name,
                                 example_code,
                                 cls_token_at_end=False,
                                 pad_on_left=False,
                                 cls_token='[CLS]',
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_token=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 cls_token_segment_id=1,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 is_eval=False,
                                 fit_to_max_corpus_len=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    # Initialize containers
    prefix_cache = list()
    target_cache = list()

    # Assign labels
    label_map = {label: i for i, label in enumerate(label_list)}

    # Obtain features
    lines = list()
    for (ex_index, example) in enumerate(examples):
        if ex_index % 1000 == 0:
            logger.info('Writing example %d of %d' % (ex_index, len(examples)))

        if example_code[0] != 'action':
            # Tokenize (used for fine-tuning on Moral Stories)
            tokens_norm = tokenizer.tokenize(example.norm) if example.norm is not None else None
            tokens_situation = tokenizer.tokenize(example.situation) if example.situation is not None else None
            tokens_intention = tokenizer.tokenize(example.intention) if example.intention is not None else None

            tokens_moral_action = tokenizer.tokenize(example.moral_action) if example.moral_action is not None else None
            tokens_immoral_action = tokenizer.tokenize(example.immoral_action) if example.immoral_action is not None else None

            tokens_moral_consequence = \
                tokenizer.tokenize(example.moral_consequence) if example.moral_consequence is not None else None
            tokens_immoral_consequence = \
                tokenizer.tokenize(example.immoral_consequence) if example.immoral_consequence is not None else None

            # Special
            tokens_moral_action_draft = tokenizer.tokenize(example.moral_action_draft) if \
                example.moral_action_draft is not None else None
            tokens_immoral_action_draft = tokenizer.tokenize(example.immoral_action_draft) if \
                example.immoral_action_draft is not None else None
            tokens_moral_consequence_draft = tokenizer.tokenize(example.moral_consequence_draft) if \
                example.moral_consequence_draft is not None else None
            tokens_immoral_consequence_draft = tokenizer.tokenize(example.immoral_consequence_draft) if \
                example.immoral_consequence_draft is not None else None

            tokens_map = {
                'nrm': tokens_norm,
                'sit': tokens_situation,
                'itn': tokens_intention,
                'ma': tokens_moral_action,
                'ia': tokens_immoral_action,
                'mc': tokens_moral_consequence,
                'ic': tokens_immoral_consequence,
                'mad': tokens_moral_action_draft,
                'iad': tokens_immoral_action_draft,
                'mcd': tokens_moral_consequence_draft,
                'icd': tokens_immoral_consequence_draft
            }
        else:
            # Tokenize (used for pre-training on Social-Chem-101)
            tokens_action = tokenizer.tokenize(example.action)
            tokens_map = {'action': tokens_action}

        # Assemble example contents according to the example code
        example_tokens = [(ec, tokens_map.get(ec, None)) for ec in example_code]
        example_tokens = [et for et in example_tokens if et[1] is not None]

        if 'gen' in task_name:
            # Add special tokens
            coded_example_tokens = list()
            active_codes = [et[0] for et in example_tokens]
            for et in example_tokens:
                if et[0] == 'ma' and 'ia' not in active_codes:
                    curr_tokens = [SPECIAL_GEN_TOKENS['action']] + et[1]
                elif et[0] == 'ia' and 'ma' not in active_codes:
                    curr_tokens = [SPECIAL_GEN_TOKENS['action']] + et[1]
                elif et[0] == 'mc' and 'ic' not in active_codes:
                    curr_tokens = [SPECIAL_GEN_TOKENS['consequence']] + et[1]
                elif et[0] == 'ic' and 'mc' not in active_codes:
                    curr_tokens = [SPECIAL_GEN_TOKENS['consequence']] + et[1]
                else:
                    curr_tokens = [SPECIAL_GEN_TOKENS[et[0]]] + et[1]
                coded_example_tokens.append(curr_tokens)
            example_tokens = coded_example_tokens
        else:
            # Remove codes
            example_tokens = [et[1] for et in example_tokens]
        lines.append((example_tokens, label_map[example.label]))

    for example_tokens, label_id in lines:
        ss_special_tokens_count = 2
        ms_special_tokens_count = 4 if sep_token_extra else 3
        # Truncate inputs, if needed
        if not fit_to_max_corpus_len:
            if len(example_tokens) > 1:
                _truncate_seq_pair(example_tokens, max_seq_length - ms_special_tokens_count - 1, 'gen' in task_name)
            else:
                if len(example_tokens[0]) > max_seq_length - ss_special_tokens_count:
                    example_tokens[0] = example_tokens[0][:(max_seq_length - ss_special_tokens_count)]

        # Construct segments
        example_prefix_length = 0
        target_ids, target_tokens, gen_prompt = list(), list(), ''

        if 'gen' not in task_name:
            tokens_a = list()
            if len(example_tokens) == 1:
                tokens_a = example_tokens[0]
            else:
                for et in example_tokens[:-1]:
                    tokens_a += et

            tokens_b = example_tokens[-1] if len(example_tokens) > 1 else None

            # Add special tokens
            tokens = tokens_a + [sep_token]
            if sep_token_extra and len(example_tokens) > 1:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if tokens_b:
                tokens += tokens_b + [sep_token]
                segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

            if cls_token_at_end:
                prefix_tokens = tokens + [cls_token]
                segment_ids = segment_ids + [cls_token_segment_id]
            else:
                prefix_tokens = [cls_token] + tokens
                segment_ids = [cls_token_segment_id] + segment_ids

            target_ids = label_id

        else:
            # For generation with GPT2 / BART / T5
            prefix_tokens = list()
            for et in example_tokens[:-1]:
                prefix_tokens += et
            target_tokens = example_tokens[-1][1:]

            # Append special tokens to generation prefixes
            if task_name in ['action|context_gen', 'action|context+consequence_gen',
                             'action|story_genref', 'action|story+consequence_genref']:
                if label_id == 0:
                    prefix_tokens += [SPECIAL_GEN_TOKENS['ia']]
                else:
                    prefix_tokens += [SPECIAL_GEN_TOKENS['ma']]
            if task_name in ['consequence|action_gen', 'consequence|action+context_gen',
                             'consequence|action+context_genref']:
                prefix_tokens += [SPECIAL_GEN_TOKENS['consequence']]
            if task_name in ['norm|actions_gen', 'norm|actions+context_gen', 'norm|actions+context+consequences_gen']:
                prefix_tokens += [SPECIAL_GEN_TOKENS['nrm']]
            example_prefix_length = len(prefix_tokens)

            # Construct model-specific input and taret sequences
            if model_name == 'gpt2':
                target_tokens = prefix_tokens + target_tokens + ['<|endoftext|>']
                if not is_eval:
                    prefix_tokens = target_tokens

            if model_name in ['t5', 'bart']:
                # Prepend special tokens to generation suffixes
                if task_name in ['action|context_gen', 'action|context+consequence_gen',
                                 'action|story_genref', 'action|story+consequence_genref']:
                    if label_id == 0:
                        target_tokens = [SPECIAL_GEN_TOKENS['ia']] + target_tokens
                        gen_prompt = SPECIAL_GEN_TOKENS['ia']
                    else:
                        target_tokens = [SPECIAL_GEN_TOKENS['ma']] + target_tokens
                        gen_prompt = SPECIAL_GEN_TOKENS['ma']
                if task_name in ['consequence|action_gen', 'consequence|action+context_gen',
                                 'consequence|action+context_genref']:
                    target_tokens = [SPECIAL_GEN_TOKENS['consequence']] + target_tokens
                    gen_prompt = SPECIAL_GEN_TOKENS['consequence']
                if task_name in ['norm|actions_gen', 'norm|actions+context_gen',
                                 'norm|actions+context+consequences_gen']:
                    target_tokens = [SPECIAL_GEN_TOKENS['nrm']] + target_tokens
                    gen_prompt = SPECIAL_GEN_TOKENS['nrm']
                target_tokens += ['</s>']  # add EOS token

            # Add dummy segment IDs
            segment_ids = [sequence_a_segment_id] * len(prefix_tokens)

            # Convert target sequence to ids
            target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

        # Convert tokens to ids
        prefix_ids = tokenizer.convert_tokens_to_ids(prefix_tokens)

        # Cache
        prefix_cache.append((prefix_ids, segment_ids, prefix_tokens, example_prefix_length))
        target_cache.append((target_ids, target_tokens, gen_prompt, tokenizer.convert_tokens_to_ids(gen_prompt)))

    # Determine maximum input tokens length
    prefix_lengths = [len(tpl[0]) for tpl in prefix_cache]
    max_prefix_length = max(prefix_lengths)
    if fit_to_max_corpus_len:
        max_seq_length = max_prefix_length
    target_lengths = list()
    if 'gen' in task_name:
        target_lengths = [len(tpl[0]) for tpl in target_cache]
        max_target_length = max(target_lengths)
        if fit_to_max_corpus_len:
            max_tgt_length = max_target_length

    # Make masks and pad inputs / labels
    features = list()
    for iid, inputs in enumerate(prefix_cache):
        example = examples[iid]
        prefix_ids, segment_ids, prefix_tokens, example_prefix_length = inputs
        target_ids, target_tokens, gen_prompt, gen_prompt_id = target_cache[iid]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        prefix_mask = [1 if mask_padding_with_zero else 0] * len(prefix_ids)

        # Zero-pad up to the sequence length
        padding_length = max_seq_length - len(prefix_ids)
        if pad_on_left:
            prefix_ids = ([pad_token] * padding_length) + prefix_ids
            prefix_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + prefix_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            prefix_ids = prefix_ids + ([pad_token] * padding_length)
            prefix_mask = prefix_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        target_mask = None
        if 'gen' in task_name:
            # Mask out padding positions, so that they don't contribute to the loss calculation
            target_padding_length = max_tgt_length - len(target_ids)
            target_mask = [1 if mask_padding_with_zero else 0] * len(target_ids)

            if pad_on_left:
                target_ids = ([-100] * target_padding_length) + target_ids
                target_mask = ([0 if mask_padding_with_zero else 1] * target_padding_length) + target_mask
            else:
                target_ids = target_ids + ([-100] * target_padding_length)
                target_mask = target_mask + ([0 if mask_padding_with_zero else 1] * target_padding_length)
            if 'gpt2' in model_name:
                # Also mask out the generation prefix (i.e. loss will only be computed over the target)
                target_ids[:example_prefix_length] = [-100] * example_prefix_length
                target_mask[:example_prefix_length] = [0] * example_prefix_length

        try:
            assert len(prefix_ids) == max_seq_length
            assert len(prefix_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
        except AssertionError:
            logging.info(prefix_ids, len(prefix_ids))
            logging.info(prefix_mask, len(prefix_mask))
            logging.info(segment_ids, len(segment_ids))
            raise AssertionError

        if iid < 5:
            logger.info('*** Example ***')
            logger.info('guid: %s' % example.guid)
            logger.info('input_tokens: %s' % ' '.join([str(x) for x in prefix_tokens]))
            logger.info('input_ids: %s' % ' '.join([str(x) for x in prefix_ids]))
            logger.info('input_mask: %s' % ' '.join([str(x) for x in prefix_mask]))
            logger.info('segment_ids: %s' % ' '.join([str(x) for x in segment_ids]))
            if 'gen' in task_name:
                logger.info('target_tokens: %s' % ' '.join([str(x) for x in target_tokens]))
                logger.info('target_ids: %s' % ' '.join([str(x) for x in target_ids]))
                logger.info('target_mask: %s' % ' '.join([str(x) for x in target_mask]))
                logger.info('gen_prompt: %s' % gen_prompt)
                logger.info('gen_prompt_ids: %s' % str(gen_prompt_id))
            else:
                logger.info('label: %s (id = %d)' % (example.label, target_ids))

        features.append(
            InputFeatures(input_ids=prefix_ids,
                          input_mask=prefix_mask,
                          segment_ids=segment_ids,
                          label_ids=target_ids,
                          label_mask=target_mask,
                          gen_prompt_id=[gen_prompt_id]))

    # Report some basic statistics
    logger.info('=' * 20)
    logger.info('Dataset statistics (before truncation / padding):')
    logger.info('Mean model input length: {:.2f}'.format(np.mean(prefix_lengths)))
    logger.info('Model input length std.: {:.2f}'.format(np.std(prefix_lengths)))
    logger.info('Min model input length: {:.2f}'.format(min(prefix_lengths)))
    logger.info('Max model input length: {:.2f}'.format(max(prefix_lengths)))
    if 'gen' in task_name:
        logger.info('-' * 20)
        logger.info('Mean model target length: {:.2f}'.format(np.mean(target_lengths)))
        logger.info('Model target length std.: {:.2f}'.format(np.std(target_lengths)))
        logger.info('Min model target length: {:.2f}'.format(min(target_lengths)))
        logger.info('Max model target length: {:.2f}'.format(max(target_lengths)))
    logger.info('=' * 20)

    return features


def _truncate_seq_pair(all_segments, max_length, is_gen):
    """ Truncates a sequence pair in place to the maximum length. """

    final_segment = list()
    if is_gen:
        # Don't truncate the target tokens
        final_segment = [all_segments[-1]]
        all_segments = all_segments[:-1]

    while True:
        total_length = sum([len(seg) for seg in all_segments])
        if total_length <= max_length:
            break
        # Shorten the longest segment
        longest_seg = all_segments[max(enumerate(all_segments), key=lambda x: len(x[1]))[0]]
        longest_seg.pop()

    all_segments += final_segment


def set_seed(args):
    """ Sets the seed to support reproducibility. """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    """ Keep a maximum of args.save_total_limit checkpoints. """
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = list()
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info('Deleting older checkpoint [{}] due to args.save_total_limit'.format(checkpoint))
        shutil.rmtree(checkpoint)


def get_token_loss(args, lm_logits, target_ids, target_mask, model_type=None):
    """ Compute token-level loss per batch during evaluation. """
    # Declare loss function
    loss_fct = CrossEntropyLoss(reduction='none')
    if model_type is None:
        model_type = args.model_type

    if model_type in ['t5', 'bart']:
        # Obtain logits to compute token-level loss / perplexity
        batch_size, max_length, vocab_size = lm_logits.shape

        # Compute loss for each instance and each token
        lm_logits = lm_logits.view(-1, vocab_size)
        lm_labels = target_ids[:, 1:].clone().contiguous().view(-1)
        token_loss = loss_fct(lm_logits, lm_labels).view(batch_size, max_length)

    else:
        # Obtain logits to compute token-level loss / perplexity
        shift_logits = lm_logits[..., :-1, :].contiguous()
        batch_size, max_length, vocab_size = shift_logits.shape

        # Compute loss for each instance and each token
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = target_ids[..., 1:].contiguous().view(-1)
        token_loss = loss_fct(shift_logits, shift_labels).view(batch_size, max_length)

    # Only consider non padded tokens
    target_mask = target_mask[..., :-1].contiguous()
    masked_token_loss = torch.mul(target_mask, token_loss)  # [batch_size, max_length]

    return masked_token_loss


## METRICS
def simple_accuracy(preds, labels):
    """ Computes prediction accuracy. """
    return np.mean(preds == labels)


def simple_f1(preds, labels):
    """ Computes prediction F1 score. """
    f1 = f1_score(y_true=labels, y_pred=preds)
    return f1


def compute_cls_metrics(preds, labels):
    """ Aggregates classification metrics. """
    assert len(preds) == len(labels)
    return {'acc': simple_accuracy(preds, labels),
            'f1': simple_f1(preds, labels)}


def compute_bleu(preds, targets):
    """ Computes corpus-level BLEU for the generated sequences. """
    targets = [targets]
    bleu = sacrebleu.corpus_bleu(preds, targets)
    return bleu.score


def compute_rouge(preds, targets):
    """ Computes ROUGE-L for the generated sequences. """
    rouge = Rouge(compute_rouge_l=True)
    rouge_out = rouge.evaluate(preds, [[tgt] for tgt in targets])
    return rouge_out[0]['rouge-l']['f1']


def compute_gen_metrics(preds, targets):
    """ Aggregates generation metrics. """
    assert len(preds) == len(targets)
    return {'BLEU-4': str(compute_bleu(preds, targets)),
            'ROUGE-L': str(compute_rouge(preds, targets))}
