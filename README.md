# AvaTaR: Optimizing LLM Agents for Tool-Assisted Knowledge Retrieval (NeurIPS 2024)

[Arxiv preprint](https://arxiv.org/abs/2406.11200) | [DSPy Implementation](https://github.com/stanfordnlp/dspy/blob/main/examples/agents/avatar_langchain_tools.ipynb)

AvaTaR is a novel and automatic framework that optimizes an LLM agent to effectively use the provided tools and improve its performance on a given task/domain. During optimization, we design a comparator module to iteratively provide insightful and holistic prompts to the LLM agent via reasoning between positive and negative examples sampled from training data.

## News

[July 2024] 🔥 Avatar is integrated into [DSPy](https://github.com/stanfordnlp/dspy) - Credit to Herumb Shandilya! You can try out [the example on jupyter notebook](https://github.com/stanfordnlp/dspy/blob/main/examples/agents/avatar_langchain_tools.ipynb). 

## 1. (For general QA）Using Avatar with DSPy

Avatar is now integrated with DSPy as `Avatar` Module for agent execution and `AvatarOptimizer` for Actor optimization. To use Avatar you'll need: Task Signature and Tools. 

* Task Signature is a `dspy.Signature` class defining the structure of your task. So if your task is of QA type you can create a signature with `question` input field and `answer` output field.
* Tools is a list of `dspy.Tools` containing all the tools of langchain tool format.

Here is an example

```python
from dspy.predict.avatar import Tool, Avatar
from langchain_community.utilities import GoogleSerperAPIWrapper, ArxivAPIWrapper

tools = [
    Tool(
        tool=GoogleSerperAPIWrapper(),
        name="WEB_SEARCH",
        desc="If you have a question, you can use this tool to search the web for the answer."
    ),
]

agent = Avatar(
    tools=tools,
    signature="question->answer",
    verbose=True,
)
```

You can execute it like any other DSPy module by passing the inputs you specified in your task signature:

```python
answer = agent(question)
```

You can optimize the Actor for optimal tool usage using `AvatarOptimizer` which optimizes it using the comparator module:

```python
from dspy.teleprompt import AvatarOptimizer

def metric(example, prediction, trace=None):
    ...

teleprompter = AvatarOptimizer(
    metric=metric,
    max_iters=10,
    max_negative_inputs=10,
    max_positive_inputs=10,
)

optimized_arxiv_agent = teleprompter.compile(
    student=agent,
    trainset=trainset
)
```

For a detailed walkthrough, you can refer to the [notebook](https://github.com/stanfordnlp/dspy/blob/avatar-optimization-integration/examples/agents/avatar_langchain_tools.ipynb) in DSPy repo.

## 2. (To reproduce the results) Run AvaTaR on STaRK and Flickr-30kEntities
### Installation

```
conda create -n avatar python=3.11
pip install stark-qa typeguard
```

### Preparation
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

### Run Agents
We already include the VSS results locally under `output/eval` and the grouping (for STaRK only) under `output/agent`. With these files, you should be able to optimize actor actions directly following the AvaTaR pipeline.

- Optimization: Following the default settings at `config/default_args.json`, run the following command to optimize the actor actions for a group of queries:
  ```bash
  sh scripts/run_avatar_stark.sh
  ```
  You can specify the dataset name and group in `scripts/run_avatar_stark.sh`. 
  ```bash
  sh scripts/run_avatar_flickr30k_entities.sh
  ```
- Evaluation: Run the following command to evaluate the optimized actor actions:
  ```bash
  sh scripts/eval_avatar_stark.sh
  ```
  or
  ```bash
  sh scripts/eval_avatar_flickr30k_entities.sh
  ```
### Run ReAct baseline
We provide the implementation of ReAct baseline on STaRK and Flickr-30kEntities. The function lists provided to ReAct are under `avatar/tools/react`. 
- Evaluation: Run the following command to evaluate ReAct:
  ```bash
  sh scripts/eval_react_stark.sh
  ```
  or
  ```bash
  sh scripts/eval_react_flickr30k_entities.sh
  ```
By default, we store the logs of ReAct reasoning and acting process at `logs/`.

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
    booktitle    = {NeurIPS},
    year         = {2024}
}
```
