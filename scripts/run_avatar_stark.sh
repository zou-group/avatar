DATASET=prime
GROUP=0

CUDA_VISIBLE_DEVICES=1,2 python scripts/run_avatar_optimizer.py \
    --dataset $DATASET \
    --group_idx $GROUP \
    --emb_model text-embedding-ada-002 \
    --agent_llm gpt-4o \
    --api_func_llm gpt-4o  \
    --use_group