from .clip import GetCLIPImageEmbedding, GetCLIPTextEmbedding
from .compute_metrics import ComputeExactMatchDirect, ComputeExactMatchScore, ComputeF1Direct, ComputeF1Score, ComputeRecallDirect, ComputeRecallScore
from .debug_print import Print2File
from .embedding import ComputeCosineSimilarity, ComputeQueryNodeSimilarity, ComputeSimilarity, GetNodeEmbedding, GetTextEmbedding
from .flickr30k_entities import GetBagOfPhrases, GetFlickrTextInfo, GetImagePatchByPhraseId, GetImages, GetPatchID2PhraseDict
from .llm_funcs import LLMCheck, LLMClassification, LLMClassifyNode, LLMExtractInfo, LLMScore, LLMSummarize, LLMVQA, LLMVisualAttribute
from .nodes import GetNodeIDs, GetNodeType
from .parser import QueryParser
from .relational_extraction import GetRelatedNodes, GetRelationDict, GetRelationTypes
from .text_extraction import GetFullInfo, GetRelevantChunk, GetRelationInfo, GetTextInfo


assigned_funcs = {
    "stark": [
        "debug_print",
        "parse_query",
        "get_node_ids_by_type",
        "get_node_type_by_id",
        "get_full_info",
        "get_text_info",
        "get_relation_info",
        "get_relevant_chunk",
        "get_text_embedding",
        "get_node_embedding",
        "get_relation_dict",
        "get_related_nodes",
        "compute_query_node_similarity",
        "compute_exact_match_score",
        "compute_cosine_similarity",
        "compute_recall_score",
        "summarize_texts_by_llm",
        "classify_by_llm", 
        "extract_relevant_info_by_llm",
        "check_req_by_llm",
        "get_scores_by_llm"
    ],
    "flickr30k_entities": [
        "debug_print",
        "parse_query",
        "get_images",
        "get_bag_of_phrases",
        "get_text_info",
        "get_clip_text_embedding",
        "get_clip_image_embedding",
        "compute_cosine_similarity",
        "get_patch_id_to_phrase_dict",
        "get_image_patch_by_phrase_id",
        "compute_f1",
        "compute_recall",
        "compute_exact_match",
        "vqa_by_llm",
        "extract_visual_attributes_by_llm"
    ]
}

general_funcs = {
    'debug_print': Print2File,
    'parse_query': QueryParser,
    'get_node_ids_by_type': GetNodeIDs,
    'get_node_type_by_id': GetNodeType,
    'get_text_info': GetTextInfo,
    'get_relation_info': GetRelationInfo,
    'get_full_info': GetFullInfo,
    'get_text_embedding': GetTextEmbedding,
    'get_clip_image_embedding': GetCLIPImageEmbedding,
    'get_clip_text_embedding': GetCLIPTextEmbedding,
    'get_node_embedding': GetNodeEmbedding, 
    'get_relevant_chunk': GetRelevantChunk,
    'get_relation_types': GetRelationTypes,
    'get_relation_dict': GetRelationDict,
    'get_related_nodes': GetRelatedNodes,
    'compute_similarity': ComputeSimilarity,
    'compute_cosine_similarity': ComputeCosineSimilarity,
    'compute_query_node_similarity': ComputeQueryNodeSimilarity,
    'compute_exact_match_score': ComputeExactMatchScore,
    'compute_recall_score': ComputeRecallScore,
    'compute_f1_score': ComputeF1Score,
    'compute_exact_match': ComputeExactMatchDirect,
    'compute_recall': ComputeRecallDirect,
    'compute_f1': ComputeF1Direct,
    'summarize_texts_by_llm': LLMSummarize,
    'extract_relevant_info_by_llm': LLMExtractInfo,
    'check_req_by_llm': LLMCheck,
    'get_scores_by_llm': LLMScore,
    'classify_by_llm': LLMClassification,
    'classify_nodes_by_llm': LLMClassifyNode,
    'vqa_by_llm': LLMVQA,
    'extract_visual_attributes_by_llm': LLMVisualAttribute
}


customized_funcs = {
    'flickr30k_entities': {
        'get_text_info': GetFlickrTextInfo,
        'get_images': GetImages,
        'get_patch_id_to_phrase_dict': GetPatchID2PhraseDict,
        'get_bag_of_phrases': GetBagOfPhrases,
        'get_image_patch_by_phrase_id': GetImagePatchByPhraseId
    }
}