1. In a retrieval test, the knowledge base contained only a short explanation of RAG, FAISS, and Groq.
2. The question "What is the annual tuition fee at KSBL?" retrieved that unrelated explanation with similarity 0.006.
3. Retrieval failed to supply supporting evidence because the indexed document contained no tuition information.
4. FAISS still returned its nearest available chunk because the search had no minimum relevance threshold.
5. One proposed fix is to calibrate a relevance threshold on validation questions and return "No relevant evidence found" below it.
