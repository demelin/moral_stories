#!/bin/bash

set -e
set -x

EXPT_FILE=./train_moral_stories_classification_beaker.yml
export MS_DATASET= ... # has to be specified
export IMAGE_NAME= ... # has to be specified
export CLUSTER= ... # has to be specified

### parameters for experiment
export MODEL_TYPE=roberta
export MODEL_NAME_OR_PATH=roberta-large

export MOUNT_POINT=/data
export OUTPUT_DIR=/output/models
export LOGGING_STEPS=500
export SAVE_STEPS=500
export SEED=42
export MAX_SEQ_LENGTH=100
export SAVE_TOTAL_LIMIT=10
export PATIENCE=10

export DATA_DIR=${MOUNT_POINT}
export GPU_COUNT=1
export GROUP_ID=$((RANDOM))

TASKS=( action_cls, action+norm_cls, action+context_cls, action+context+consequence_cls )
# TASKS=( consequence+action_cls, consequence+action+context_cls )
SPLITS=( norm_distance lexical_bias minimal_pairs )

BATCHSIZES=( 8 16 )
LEARNINGRATES=( 1e-5 3e-5 5e-5)
EPOCHS=( 3 4 )

for tsk in "${TASKS[@]}"
do
  export TASK_NAME=${tsk}
  for spl in "${SPLITS[@]}"
  do
    export SPLIT_NAME=${spl}
    for s in "${BATCHSIZES[@]}"
    do
      export TRAIN_BATCH_SIZE=${s}
      export EVAL_BATCH_SIZE=${s}
      for l in "${LEARNINGRATES[@]}"
      do
        export LEARNING_RATE=${l}
        for e in "${EPOCHS[@]}"
        do
          export NUM_EPOCHS=${e}
          export WARMUP_PCT=0.1
          export DESC="$TASK_NAME-$SPLIT_NAME-$GROUP_ID"
          expt=$(beaker experiment create --workspace YOUR_WORKSPACE -file ${EXPT_FILE})
        done
      done
    done
  done
done

echo "-->> Starting experiments with Group ID $GROUP_ID <<---"
