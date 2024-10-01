import os
import os.path as osp
from .vss import  VSS
from .llm_reranker import LLMReranker
from .multi_vss import MultiVSS
from .avatar import AvaTaR
from .llmv_reranker import LLMvReranker
from .dense_retriever import DenseRetrieval
from .react import React


def get_model(args, kb):
    model_name = args.model
    if model_name == 'VSS':
        return VSS(
            kb,
            emb_model=args.emb_model,
            query_emb_dir=args.query_emb_dir, 
            candidates_emb_dir=args.node_emb_dir
        )
    if model_name == 'MultiVSS':
        return MultiVSS(
            kb,
            emb_model=args.emb_model,
            query_emb_dir=args.query_emb_dir, 
            candidates_emb_dir=args.node_emb_dir,
            chunk_emb_dir=args.chunk_emb_dir,
            aggregate=args.aggregate,
            chunk_size=args.chunk_size,
            max_k=args.multi_vss_topk
        )
    if model_name in "LLMReranker":
        return LLMReranker(kb, 
                           emb_model=args.emb_model,
                           llm_model=args.llm_model,
                           query_emb_dir=args.query_emb_dir, 
                           candidates_emb_dir=args.node_emb_dir,
                           max_cnt = args.max_retry,
                           max_k=args.llm_topk
                           )
    if model_name == 'LLMvReranker':
        return LLMvReranker(kb, 
                            model_name=args.vlm_model,
                            query_emb_dir=args.query_emb_dir, 
                            candidates_emb_dir=args.node_emb_dir,
                            max_k=args.llm_topk
                            )
    if model_name == 'avatar':
        output_dir = osp.join(args.output_dir, 'agent', args.dataset, model_name, args.agent_llm)
        os.makedirs(name=output_dir, exist_ok=True)
        return AvaTaR(kb, 
                      emb_model=args.emb_model,
                      agent_llm=args.agent_llm,
                      api_func_llm=args.api_func_llm,
                      output_dir=output_dir,
                      chunk_size=args.chunk_size,
                      query_emb_dir=args.query_emb_dir,
                      chunk_emb_dir=args.chunk_emb_dir,
                      node_emb_dir=args.node_emb_dir,
                      topk_test=args.topk_test,
                      dataset=args.dataset,
                      num_processes=args.num_processes,
                      )
    if 'DenseRetriever' in model_name:
        return DenseRetrieval(
            kb=kb,
            model_path=args.model_path, 
            doc_enc_dir=args.doc_enc_dir,
            query_emb_dir=args.query_emb_dir, 
            candidates_emb_dir=args.node_emb_dir, 
            candidates_dir=args.candidates_dir,
            eval_batch_size=args.eval_batch_size,
            num_candidates=args.num_candidates,
            dataset=args.dataset,
            renew_candidates=False
        )
    if 'React' in model_name:
        output_dir = osp.join(args.output_dir, 'agent', args.dataset, model_name, args.agent_llm)
        os.makedirs(name=output_dir, exist_ok=True)
        return React(kb, 
                    llm_func_model=args.agent_llm,
                    emb_model=args.emb_model,
                    output_dir=output_dir,
                    chunk_size=args.chunk_size,
                    query_emb_dir=args.query_emb_dir,
                    chunk_emb_dir=args.chunk_emb_dir,
                    node_emb_dir=args.node_emb_dir,
                    n_init_candidates=args.n_init_candidates, # 20
                    dataset=args.dataset,
                    vision=args.vision,
                )
    raise NotImplementedError(f'{model_name} not implemented')