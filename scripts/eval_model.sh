#!/bin/sh

python parlai/scripts/eval_model.py \
    -t blended_skill_talk,wizard_of_wikipedia,convai2:normalized,empathetic_dialogues \
    -dt test \
    -mf ${1} \
    --n-segments 2 \
    --fixed-candidates-path ${FIXED_CANDIDATE_PATH} \
    --load-candidate-encs true \
    --load-candidate-encs-path ./fixed_candidates.encs.npy \
    --save-candidate-encs-path ./fixed_candidates.encs.npy \
    --load-candidate-vecs true \
    --load-candidate-retriever-vecs-path ./fixed_candidates_retriever.vecs.npy \
    --save-candidate-retriever-vecs-path ./fixed_candidates_retriever.vecs.npy \
    --load-candidate-generator-vecs-path ./fixed_candidates_generator.vecs.npy \
    --save-candidate-generator-vecs-path ./fixed_candidates_generator.vecs.npy \
    --batchsize 40 \
    --skip-generation True \
    --corge-topk-cands 1 \
    --model-parallel False \
