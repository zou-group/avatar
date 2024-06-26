CUDA_VISIBLE_DEVICES=4 python scripts/emb_generate.py \
    --dataset flickr30k_entities \
    --flickr30k_entities_root /dfs/project/kgrlm/benchmark \
    --emb_model openai/clip-vit-large-patch14 \
    --mode image \
    --batch_size 4

CUDA_VISIBLE_DEVICES=4 python scripts/emb_generate.py \
    --dataset flickr30k_entities \
    --flickr30k_entities_root /dfs/project/kgrlm/benchmark \
    --emb_model openai/clip-vit-large-patch14 \
    --mode query \
    --batch_size 4