CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval.py \
    --model avatar \
    --split human_generated_eval \
    --group_idx 0 \
    --dataset prime \
    --emb_model text-embedding-ada-002 \
    --agent_llm gpt-4o \
    --api_func_llm gpt-4o \
    --seed 20