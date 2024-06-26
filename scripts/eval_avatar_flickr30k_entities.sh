CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/run_avatar_optimizer.py \
    --dataset flickr30k_entities \
    --emb_model openai/clip-vit-large-patch14 \
    --agent_llm gpt-4o 