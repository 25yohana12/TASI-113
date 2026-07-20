Hybrid Text-to-SQL Framework
---
A hybrid Text-to-SQL framework integrating Retrieval-Augmented Generation (RAG), In-Context Reinforcement Learning (ICRL), and SLM-based Hierarchical Action Correction (SHARE) to improve SQL generation for complex natural language queries.
This repository contains the implementation, experimental results, and supporting materials for my undergraduate thesis.

Overview
---
Text-to-SQL systems often struggle with complex database schemas and multi-step reasoning. This research proposes a hybrid framework that combines:
RAG to retrieve relevant schema and query examples.
ICRL to iteratively refine SQL generation through contextual feedback.
SHARE to hierarchically detect and correct SQL errors using a Small Language Model.

The framework is evaluated on the BIRD Development Set using multiple evaluation metrics, including:
Execution Accuracy (EX)
Inference Latency
Token Usage
Error Distribution

Dataset
---
The experiments are conducted on the BIRD benchmark.
Due to licensing restrictions, the original dataset is not included in this repository.

