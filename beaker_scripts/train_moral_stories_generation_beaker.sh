#!/bin/bash

set -e
set -x

EXPT_FILE=./train_moral_stories_generation_beaker.yml
export MS_DATASET= ... # has to be specified
export IMAGE_NAME= ... # has to be specified
export CLUSTER= ... # has to be specified

### parameters for experiment
export MOUNT_POINT=/data
export OUTPUT_DIR=/output/models
export LOGGING_STEPS=500
export SAVE_STEPS=500
export SEED=42
export MAX_SEQ_LENGTH=100
export GRAD_ACC_STEPS=8
export SAVE_TOTAL_LIMIT=6
export PATIENCE=3

export MAX_GEN_LENGTH=60
export P=0.9

export DATA_DIR=${MOUNT_POINT}
export GPU_COUNT=1
export GROUP_ID=$((RANDOM))

# Staged tasks, uncomment ones to run
TASKS=( action|context_gen, action|context+consequence_gen )
# TASKS=( consequence|action_gen, consequence|action+context_gen )
# TASKS=( norm|actions_gen, norm|actions+context_gen, norm|actions+context+consequences_gen )
SPLITS=( norm_distance )
MODEL_TYPES=( gpt2 t5 bart )
MODEL_NAMES=( gpt2-xl t5-large facebook/bart-large )

BATCHSIZES=( 8 )
LEARNINGRATES=( 5e-6 )
EPOCHS=( 50 )  # upper limit, but early-stopping is enabled

for ((i=0;i<${#MODEL_TYPES[@]};++i));
do
  export MODEL_TYPE=${MODEL_TYPES[i]}
  export MODEL_NAME_OR_PATH=${MODEL_NAMES[i]}
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
            expt=$(beaker experiment create --workspace YOUR_WORKSPACE --file ${EXPT_FILE})
          done
        done
      done
    done
  done
done

echo "-->> Starting experiments with Group ID $GROUP_ID <<---"
