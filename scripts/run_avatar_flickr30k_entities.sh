CUDA_VISIBLE_DEVICES=7 python scripts/run_avatar_optimizer.py \
    --dataset flickr30k_entities \
    --emb_model openai/clip-vit-large-patch14 \
    --agent_llm gpt-4o \
    --api_func_llm gpt-4o 