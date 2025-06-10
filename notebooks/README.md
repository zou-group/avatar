# AvaTaR on Question Answering Tasks

These notebooks follow [dspy/blob/main/examples/agents/avatar_langchain_tools.ipynb](https://github.com/stanfordnlp/dspy/blob/avatar-optimization-integration/examples/agents/avatar_langchain_tools.ipynb) to build avatar training and testing pipeline, with minor modifications on data loading and training parameters.

To run them, please further install the following packages
```
pip install git+https://github.com/stanfordnlp/dspy.git
pip install langchain-community langchain_openai arxiv wikipedia
pip install jsonlines sentence_transformers chromadb
```

Note: Sometimes the training stuck when because of the multithreading usage in `dspy.teleprompt.AvatarOptimizer.thread_safe_evaluator(self, devset, actor, return_outputs=False, num_threads=60)`, manually changing `num_threads` to `1` would resolve this issue. 
