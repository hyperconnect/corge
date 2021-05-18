#!/bin/bash

python parlai/scripts/eval_model.py \
    -t blended_skill_talk \
    -dt test \
    -mf ${1} \
    --fixed-candidates-path ${FIXED_CANDIDATE_PATH} \
    --load-candidate-encs true \
    --load-candidate-encs-path ./fixed_candidates.encs.npy \
    --save-candidate-encs-path ./fixed_candidates.encs.npy \
    --load-candidate-vecs true \
    --load-candidate-retriever-vecs-path ./fixed_candidates_retriever.vecs.npy \
    --save-candidate-retriever-vecs-path ./fixed_candidates_retriever.vecs.npy \
    --load-candidate-generator-vecs-path ./fixed_candidates_generator.vecs.npy \
    --save-candidate-generator-vecs-path ./fixed_candidates_generator.vecs.npy \
    --inference beam \
    --beam-size 10 \
    --beam-min-length 20 \
    --n-segments 2 \
    --corge-topk-cands 1 \
    --generation-result-path ${1}-test \
    --batchsize 140 \
    --num-generated-responses 980 \
    --num-examples 980 \
    --beam-block-ngram 3 \
    --beam-block-full-context True \
    --beam-context-block-ngram 3 \
    --model-parallel False \
