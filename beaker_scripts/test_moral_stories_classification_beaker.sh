#!/bin/bash

set -e
set -x

EXPT_FILE=./test_moral_stories_classification_beaker.yml
export MS_DATASET= ... # has to be specified
export MODEL_DATASET= ... # has to be specified
export IMAGE_NAME= ... # has to be specified
export CLUSTER= ... # has to be specified

### parameters for experiment
export MOUNT_POINT=/data
export MODEL_MOUNT_POINT=/model
export MODEL_NAME_OR_PATH=/model/... # has to be specified
export OUTPUT_DIR=/output/models
export LOGGING_STEPS=500
export SAVE_STEPS=500
export SEED=42
export MAX_SEQ_LENGTH=100
export GRAD_ACC_STEPS=8
export SAVE_TOTAL_LIMIT=6

export DATA_DIR=${MOUNT_POINT}
export GPU_COUNT=1
export GROUP_ID=$((RANDOM))

TASKS=( action__story_cls  )  # for example
SPLITS=( norm_distance )  # for example
MODEL_TYPES=( roberta )

BATCHSIZES=( 8 )
LEARNINGRATES=( 1e-5 )
EPOCHS=( 1 )

export SEED=$((RANDOM))
for ((i=0;i<${#MODEL_TYPES[@]};++i));
do
  export MODEL_TYPE=${MODEL_TYPES[i]}
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
            export DESC="GEN-EVAL-$TASK_NAME-$SPLIT_NAME-$GROUP_ID"
            expt=$(beaker experiment create --workspace YOUR_WORKSPACE --file ${EXPT_FILE})
          done
        done
      done
    done
  done
done

echo "-->> Starting experiments with Group ID $GROUP_ID <<---"
