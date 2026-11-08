# RAG pipeline concepts

Retrieval-augmented generation (RAG) answers a question by first retrieving relevant passages from your own documents, then asking a language model to answer using only those passages. It lets a model work with private or recent information without retraining, and it makes answers checkable because each claim can point to a source.

## Ingestion

Ingestion runs ahead of time. Documents are split into chunks, each chunk is converted into an embedding vector, and the vectors are saved in a vector store together with the chunk text and its source name.

## Chunking

Chunk size is a trade-off. Small chunks match a question precisely but can lose surrounding context. Large chunks keep context but dilute the match and spend more tokens. A few hundred to about a thousand characters with some overlap is a common starting point. Overlap means a sentence near a boundary appears in both neighbouring chunks.

## Embeddings

An embedding model maps text to a vector so that texts with similar meaning land close together. The same model must embed both the chunks and the question. Some models, such as the BGE family, embed queries and passages slightly differently.

## Retrieval

At question time the question is embedded and the store returns the k chunks with the highest cosine similarity. Cosine similarity is the dot product of two vectors after both are scaled to unit length. Typical values of k are 3 to 8.

## Generation

The retrieved chunks are placed into the prompt as numbered sources, and the model is told to answer only from them, cite them, and say so when the sources do not contain the answer. This grounding instruction is what reduces hallucination.

## Common failure modes

Retrieval misses: the right chunk exists but is not in the top k, often because of poor chunking or a mismatch between the question's wording and the document's. Context stuffing: too many irrelevant chunks distract the model. Stale index: documents changed but were not re-ingested. Improvements include hybrid search (keyword plus vector), re-ranking the top results with a cross-encoder, and query rewriting.
