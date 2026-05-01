## Project: Document Retrieval 

1. In this project, you will implement basic retrieval methods and evaluate their performance. It does not provide initial codebase since the implementation is relatively easy. Please see the references (like [5]) for example implementation.

### Your task
1. Implement different retrieval methods:
   (1) Sparse retrieval:
       a. TF-IDF
       b. BM25 
   (2) Dense retrieval:
       You can use any state-of-the-art model to generate document embeddings, then I recommend using FAISS[6] for KNN search.  
   (3) Sparse + dense retrieval:
       Combine sparse and dense retrieval to achieve higher quality. You can use any way to combine them, including but not limited to:
       a. Intersecting the results
       b. Unioning the results
       c. Ranking the results by weighted summation of the scores as in [1]
       d. Any other ways in related work or you come up with

3. Implement the performance metric calculation:
   (1) Mean Reciprocal Rank (MRR) for measuring quality
   (2) Retrieval time for measuring speed

4. Evaluate factors that may impact the performance of the methods you implement, including but not limited to:
   (1) Different embedding models - see [2] for more details
       a. KNRM[3]: Query-document cross encoder model (Figure 2b in [2]).
       a. Sentence-BERT[4]: Two-tower model (Figure 2a in [2])
       b. ColBERT[2]: Late interaction model (Figure 2d in [2])
   (2) Data scale
   (3) Different ways to combine sparse and dense retrieval
   (4) Different distance/similarity metrics
   (5) Any other factors you think will have influence

   **Note**: for each point above, you should not only report the evaluation results, but also analyze why.

5. Propose a few potential research directions, the reasons to choose them, and your solutions (no need to be implemented).  

### Datasets:
You can use the passage ranking dataset of **MS MARCO**[7], a popular retrieval benchmark, to conduct your evaluation. See [5] for an example. Feel free to explore other materials in [5] which are useful.  

### References
[1] https://arxiv.org/abs/2401.04055
[2] ColBERT: https://arxiv.org/pdf/2004.12832
[3] KNRM: https://github.com/AdeDZY/K-NRM
[4] Sentence-BERT: https://arxiv.org/abs/1908.10084
[5] https://github.com/UKPLab/sentence-transformers/tree/master/examples/sentence_transformer/training/ms_marco
[6] FAISS: https://github.com/facebookresearch/faiss/wiki
[7] https://microsoft.github.io/msmarco/
