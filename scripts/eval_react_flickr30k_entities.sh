mkdir -p logs/
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval.py \
    --model React \
    --split human_generated_eval \
    --group_idx 0 \
    --dataset flickr30k_entities \
    --emb_model text-embedding-ada-002 \
    --agent_llm claude-3-opus-20240229 \
    --api_func_llm claude-3-opus-20240229 \
    --seed 20 \
    --vision True
> logs/eval_react_flickr30k_entities.log