#!/bin/sh

model_file=${1}/corge90m-numcand${2}-bsz${3}-lr${4}
corge_num_cands=${2}
batch_size=${3}
lr=${4}

python parlai/scripts/train_model.py \
    -t blended_skill_talk,wizard_of_wikipedia,convai2:normalized,empathetic_dialogues \
    --multitask-weights 1,3,3,3 \
    -veps 0.25 \
    --attention-dropout 0.0 \
    --batchsize ${batch_size} \
    --model transformer/corge \
    --history-add-global-end-token end \
    --label-truncate 128 \
    -lr ${lr} \
    --lr-scheduler reduceonplateau \
    --lr-scheduler-patience 3 \
    --optimizer adam \
    --model-parallel true \
    --save-after-valid true \
    --text-truncate 128 \
    --truncate 128 \
    --warmup_updates 100 \
    --fp16 false \
    --fp16-impl mem_efficient \
    --update-freq 2 \
    --gradient-clip 0.1 \
    --skip-generation false \
    --model-file ${model_file} \
    --retriever-model-path ${RETRIEVER_MODEL_PATH} \
    --generator-model-path ${GENERATOR_MODEL_PATH} \
    --tensorboard-log true \
    \
    `# tokenizer params` \
    --dict-file ${GENERATOR_MODEL_PATH}.dict \
    --dict-tokenizer bpe \
    --dict-lower true \
    \
    `# validation` \
    --eval-batchsize ${batch_size} \
    -veps 0.25 \
    -vme 10000 \
    -vp 10 \
    --validation-metric ppl \
    --validation-metric-mode min \
    --save-after-valid true \
    --log-every-n-secs 20 \
    \
    `# miscs` \
    --init-corge-training \
    --corge-topk-cands ${corge_num_cands} \
    --fixed-candidates-path ${FIXED_CANDIDATE_PATH} \
    --load-candidate-encs true \
    --load-candidate-encs-path ./fixed_candidates.encs.npy \
    --save-candidate-encs-path ./fixed_candidates.encs.npy \
    --load-candidate-vecs true \
    --load-candidate-retriever-vecs-path ./fixed_candidates_retriever.vecs.npy \
    --save-candidate-retriever-vecs-path ./fixed_candidates_retriever.vecs.npy \
    --load-candidate-generator-vecs-path ./fixed_candidates_generator.vecs.npy \
    --save-candidate-generator-vecs-path ./fixed_candidates_generator.vecs.npy \
    --n-segments 2 \
    --inference beam \
    --beam-min-length 10 \
    --beam-size 5 \
    --skip-generation false \
    --save-generated-samples True \
    --generation-result-path ${model_file}-inference \
    --num-generated-responses 100 \
