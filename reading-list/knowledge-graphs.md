# Knowledge graphs — external links

Curated reading list for knowledge graphs and GraphRAG. Unread; working through these.

## Courses

- [Knowledge Graphs for RAG](https://www.deeplearning.ai/courses/knowledge-graphs-rag) — DeepLearning.AI, Andreas Kollegger (Neo4j). 9 lessons, ~2h. Nodes/edges fundamentals, Cypher, adding a vector index to an existing graph, building a KG from SEC 10-K filings, connecting multiple graphs, then LangChain QA over the result. Assumes some LangChain familiarity.
  - Notes: [kaushikacharya/Knowledge_Graphs_for_RAG](https://github.com/kaushikacharya/Knowledge_Graphs_for_RAG) — third-party per-lesson write-ups (`notes/`, lessons 0-7) plus `code/` assignments. Good for skimming the course content without sitting through the videos.

## GraphRAG walkthroughs

- [Building, Improving, and Deploying Knowledge Graph RAG Systems on Databricks](https://www.databricks.com/blog/building-improving-and-deploying-knowledge-graph-rag-systems-databricks) — Databricks Blog. GraphRAG with Neo4j as the graph store, driven from Databricks: Delta Tables as the source, LLM-generated Cypher for retrieval, HNSW vector indexes, deployment via Agent Bricks Custom Agents. Core argument is combining explicit graph relationships with implicit embedding similarity.
- [RAG with knowledge graphs (Neo4j)](https://huggingface.co/learn/cookbook/en/rag_with_knowledge_graphs_neo4j) — HuggingFace Cookbook. Runnable notebook over synthetic research-article/author data. Puts both retrieval modes side by side, vector similarity and Cypher graph traversal, wired through LangChain, to answer collaboration- and topic-style questions.
- [Neo4j RAG tutorial](https://neo4j.com/blog/developer/rag-tutorial/) — Neo4j developer blog.
