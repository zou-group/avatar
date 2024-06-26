CUDA_VISIBLE_DEVICES=0 python scripts/emb_generate.py --dataset amazon --emb_model text-embedding-ada-002 --mode query
CUDA_VISIBLE_DEVICES=0 python scripts/emb_generate.py --dataset amazon --emb_model text-embedding-ada-002 --mode doc
CUDA_VISIBLE_DEVICES=0 python scripts/emb_generate.py --dataset mag --emb_model text-embedding-ada-002 --mode query
CUDA_VISIBLE_DEVICES=0 python scripts/emb_generate.py --dataset mag --emb_model text-embedding-ada-002 --mode doc
CUDA_VISIBLE_DEVICES=0 python scripts/emb_generate.py --dataset prime --emb_model text-embedding-ada-002 --mode query
CUDA_VISIBLE_DEVICES=0 python scripts/emb_generate.py --dataset prime --emb_model text-embedding-ada-002 --mode doc