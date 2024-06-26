# AvaTaR: Optimizing LLM Agents for Tool-Assisted Knowledge Retrieval

Paper: [Arxiv preprint](https://arxiv.org/abs/2406.11200)

AvaTaR is a novel and automatic framework that optimizes an LLM agent to effectively use the provided tools and improve its performance on a given task/domain. During optimization, we design a comparator module to iteratively provide insightful and holistic prompts to the LLM agent via reasoning between positive and negative examples sampled from training data.

## Installation

```
conda create -n avatar python=3.11
pip install stark-qa typeguard
```

## Preparation
- Specify API keys in command line
    ```bash
    export ANTHROPIC_API_KEY=YOUR_API_KEY
    ```
    ```bash
    export OPENAI_API_KEY=YOUR_API_KEY
    export OPENAI_ORG=YOUR_ORGANIZATION
    ```
- Embeddings: Download all embeddings by running the following script:
  ```bash
  sh scripts/emb_download_all.sh
  ```
- Raw data:
  STaRK data will be downloaded automatically when running the code. 
  For Flickr30k Entities, submit form at [Flickr 30k & Denotation Graph data](https://forms.illinois.edu/sec/229675) to request access. Then organize the data as follows:
  ```
  data
  ├── flickr30k_entities
  │   ├── raw
  │   │   ├── Annotations
  │   │   │   ├── 36979.xml
  │   │   │   ├── ...
  │   │   ├── flickr30k-images
  │   │       ├── 36979.jpg
  │   │       ├── ...
  │   ├── split
  │   │   ├── test.index
  │   │   ├── train.index
  │   │   ├── val.index
  │   ├── qa.csv
  ├── ...
  ```

## Run Agents
We already include the VSS results locally under `output/eval` and the grouping (for STaRK only) under `output/agent`. With these files, you should be able to optimize actor actions directly following the AvaTaR pipeline.

- Optimization: Following the default settings at `config/default_args.json`, run the following command to optimize the actor actions for a group of queries:
  ```bash
  sh scripts/run_avatar_stark.sh
  ```
  You can specify the dataset name and group in `scripts/run_avatar_stark.sh`. 
  ```bash
  sh run_avatar_flickr30k_entities.sh
  ```
- Evaluation: Run the following command to evaluate the optimized actor actions:
  ```bash
  sh scripts/run_eval_avatar_stark.sh
  ```
  or
  ```bash
  sh scripts/run_eval_avatar_flickr30k_entities.sh
  ```
## Reference 

```
@article{wu24avatar,
    title        = {AvaTaR: Optimizing LLM Agents for Tool-Assisted Knowledge Retrieval},
    author       = {
        Shirley Wu and Shiyu Zhao and 
        Qian Huang and Kexin Huang and 
        Michihiro Yasunaga and Kaidi Cao and 
        Vassilis N. Ioannidis and Karthik Subbian and 
        Jure Leskove and James Zou
    },
    eprinttype   = {arXiv},
    eprint       = {2406.11200},
  year           = {2024}
}
```
