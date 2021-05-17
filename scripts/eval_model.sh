#!/bin/sh

python parlai/scripts/eval_model.py \
    -t blended_skill_talk,wizard_of_wikipedia,convai2:normalized,empathetic_dialogues \
    -dt test \
    -mf ${1} \
    --n-segments 2 \
    --fixed-candidates-path ${FIXED_CANDIDATE_PATH} \
    --batchsize 40 \
    --skip-generation True \
    --corge-topk-cands 1 \
    --model-parallel False \
