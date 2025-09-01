<!-- omit in toc -->
# RAG with RAPTOR

There are [tons of RAG optimization techniques](https://levelup.gitconnected.com/testing-18-rag-techniques-to-find-the-best-094d166af27f) you can use to improve performance, from query transformations to sophisticated re-ranking models. The challenge is that each new layer often brings added complexity, more LLM calls, and more moving parts to your architecture.

> But what if we could get a better performance by focusing on just one thing: building a smarter index?

![Raptor with RAG](https://miro.medium.com/v2/resize:fit:1250/1*MP4ZNLcEJevendLkd50jig.png)
*Raptor with RAG (Created by Fareed Khan)*

This is the core idea behind **RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)**, it keeps RAG simple at query time while delivering superior results by building a hierarchical index that mirrors human understanding from details to high-level concepts.

Here is a high-level overview of how RAPTOR works:

1.  **Start with Leaf Nodes:** First, we break down all source documents into small, detailed chunks. These are the foundational “leaf nodes” of our knowledge tree.
2.  **Cluster for Themes:** Then, we use an advanced clustering algorithm to automatically group these leaf nodes into thematically related clusters based on their semantic meaning.
3.  **Summarize for Abstraction:** We use an LLM to generate a concise, high-quality summary for each cluster. These summaries become the next, more abstract layer of the tree.
4.  **Recurse to Build Upwards:** We repeat the clustering and summarization process on the newly created summaries, building the tree level by level towards higher concepts.
5.  **Index Everything Together:** Finally, we combine all text, the original leaf chunks and all generated summaries into a single **“collapsed tree”** vector store for a powerful, multi-resolution search.

In this blog, we are going to…
> Evaluate a simple RAG pipeline against a RAPTOR-based RAG pipeline and explore why RAPTOR performs better than other approaches.

<!-- omit in toc -->
## Table of Contents
- [Initializing our RAG Configuration](#initializing-our-rag-configuration)
- [Data Ingestion and Preparation](#data-ingestion-and-preparation)
- [Creating Leaf Nodes of RAPTOR Tree](#creating-leaf-nodes-of-raptor-tree)
    - [What is the Point of Leaf Nodes?](#what-is-the-point-of-leaf-nodes)
- [Implementing a Simple RAG Approach](#implementing-a-simple-rag-approach)
- [Building a Hierarchical Clustering Engine](#building-a-hierarchical-clustering-engine)
  - [Dimensionality Reduction with UMAP](#dimensionality-reduction-with-umap)
  - [Optimal Cluster Number Detection](#optimal-cluster-number-detection)
  - [Probabilistic Clustering with GMM](#probabilistic-clustering-with-gmm)
  - [Hierarchical Clustering Orchestrator](#hierarchical-clustering-orchestrator)
- [Building and Executing the RAPTOR Tree](#building-and-executing-the-raptor-tree)
    - [The Abstraction Engine: Summarization](#the-abstraction-engine-summarization)
    - [The Recursive Tree Builder](#the-recursive-tree-builder)
- [Indexing with the Collapsed Tree Strategy](#indexing-with-the-collapsed-tree-strategy)
    - [Query 1: Specific, Low-Level Question](#query-1-specific-low-level-question)
    - [Query 2: Mid-Level, Conceptual Question](#query-2-mid-level-conceptual-question)
    - [Query 3: Broad, High-Level Question](#query-3-broad-high-level-question)
- [Quantitative Evaluation of RAPTOR against Simple RAG](#quantitative-evaluation-of-raptor-against-simple-rag)
- [Qualitative Evaluation using LLM as a Judge](#qualitative-evaluation-using-llm-as-a-judge)
- [Summarizing the RAPTOR Approach](#summarizing-the-raptor-approach)

---

## Initializing our RAG Configuration

The two most important components of any RAG system are:

1.  An embedding model → to convert documents into vector space for retrieval.
2.  A text generation model (LLM) → to interpret retrieved content and produce answers.

> To make our approach **replicable and fair**, we are intentionally using a **quantized, older model** that was released about a year ago.

If we used a **newer LLM**, it might already “know” the answers internally, bypassing retrieval. By choosing an older model, we ensure that the evaluation truly tests **retrieval quality** which is exactly where RAPTOR vs. simple RAG makes a difference.

We first need to import PyTorch and other supporting components:

```python
# Import the core PyTorch library for tensor operations
import torch

# Import LangChain's wrappers for Hugging Face models
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

# Import core components from the transformers library for model loading and configuration
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# Import LangChain's tools for prompt engineering and output handling
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
```

We will use `sentence-transformers/all-MiniLM-L6-v2`, a lightweight and widely used embedding model, to convert all our text chunks and summaries into vector representations.

```python
# --- Configure Embedding Model ---
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Use GPU if available, otherwise fallback to CPU
model_kwargs = {"device": "cuda"}

# Initialize embeddings with LangChain's wrapper
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=model_kwargs
)
```

This embedding model is small but perfect for large-scale document indexing without excessive memory usage. Next for generation, we are using [Mistral-7B-Instruct v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2), a capable but compact instruction-tuned model.

To make it memory-friendly, we load it with **4-bit quantization** using `BitsAndBytesConfig`.

```python
# --- Configure LLM for Summarization and Generation ---
llm_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Quantization: reduces memory footprint while preserving performance
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
```

We now need to load the tokenizer and the LLM itself with the quantization settings applied.

```python
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(llm_id)

# Load LLM with quantization
model = AutoModelForCausalLM.from_pretrained(
    llm_id,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config
)
```

This way, the model runs efficiently on available hardware, even with limited GPU memory. Once the model and tokenizer are loaded, we wrap them in a Hugging Face **pipeline** for text generation.

```python
# Create a text-generation pipeline using the loaded model and tokenizer.
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=512 # Controls the max length of the generated summaries and answers
)
```

Finally, we wrap the Hugging Face pipeline in LangChain’s `HuggingFacePipeline` so it integrates smoothly with our retrieval pipeline later.

```python
# Wrap pipeline for LangChain compatibility
llm = HuggingFacePipeline(pipeline=pipe)
```

---

## Data Ingestion and Preparation

To properly showcase how **RAPTOR** can improve **RAG** performance, we need a **complex and challenging database**. The idea is that when we run queries against it, we want to see real differences between **simple RAG** and **RAPTOR-enhanced RAG**.

For this reason, we are focusing on the **Hugging Face documentation**. The docs are rich in overlapping information and contain subtle variations that can easily trip up a naïve retriever.

For example, Hugging Face explains **ZeRO-3 checkpoint saving** in multiple ways:
- `trainer.save_model()`
- `unwrap_model().save_pretrained()`
- `zero_to_fp32()`

All of these refer to the same underlying concept, consolidating model shards into a full checkpoint.
> A simple RAG pipeline might retrieve only one of these variants and **miss the broader context**, leading to incomplete or even broken instructions. RAPTOR, on the other hand, can consolidate and reason across them.

Since Hugging Face has an extensive documentation ecosystem, we are narrowing down to five **core guides** where most practical usage happens. Let’s initialize their URLs.

```python
# Define the documentation sections to scrape, with varying crawl depths.
urls_to_load = [
    {"url": "https://huggingface.co/docs/transformers/index", "max_depth": 3},
    {"url": "https://huggingface.co/docs/datasets/index", "max_depth": 2},
    {"url": "https://huggingface.co/docs/tokenizers/index", "max_depth": 2},
    {"url": "https://huggingface.co/docs/peft/index", "max_depth": 1},
    {"url": "https://huggingface.co/docs/accelerate/index", "max_depth": 1}
]
```

A key parameter here is `max_depth`, which controls how deeply we crawl from the starting page.

![How the depth parameter works](https://miro.medium.com/v2/resize:fit:875/1*N-RUEgcCSsdiEg72w3KP9w.png)
*Depth parameter work (Created by Fareed Khan)*

- It starts with the root page (`...docs/transformers/index`).
- From there, it follows all links on that page → this is depth 1.
- Then, it crawls into the links found inside those subpages → this is depth 2.
- Finally, it continues one more level into the links within those sub-subpages → this is depth 3.

Now, we will fetch the content using LangChain's `RecursiveUrlLoader` with BeautifulSoup.

```python
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup

# Empty list to append components
docs = []

# Iterate through the list and crawl each documentation section.
for item in urls_to_load:
    # Initialize the loader with the specific URL and parameters.
    loader = RecursiveUrlLoader(
        url=item["url"],
        max_depth=item["max_depth"],
        extractor=lambda x: Soup(x, "html.parser").text, # Use BeautifulSoup to extract text
        prevent_outside=True, # Ensure we stay within the documentation pages
        use_async=True, # Use asynchronous requests for faster crawling
        timeout=600, # Set a generous timeout for slow pages
    )
    # Load the documents and add them to our master list.
    loaded_docs = loader.load()
    docs.extend(loaded_docs)
    print(f"Loaded {len(loaded_docs)} documents from {item['url']}")
```

Running this loop gives the following output:
```bash
###### OUTPUT #######
Loaded 68 documents from https://huggingface.co/docs/transformers/index
Loaded 35 documents from https://huggingface.co/docs/datasets/index
Loaded 21 documents from https://huggingface.co/docs/tokenizers/index
Loaded 12 documents from https://huggingface.co/docs/peft/index
Loaded 9 documents from https://huggingface.co/docs/accelerate/index

Total documents loaded: 145
```
We have a total of `145` documents. Let’s analyze their token counts.

```python
import numpy as np
import matplotlib.pyplot as plt

# We need a consistent way to count tokens, using the LLM's tokenizer is the most accurate method.
def count_tokens(text: str) -> int:
    """Counts the number of tokens in a text using the configured tokenizer."""
    # Ensure text is not None and is a string
    if not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text))

# Extract the text content from the loaded LangChain Document objects
docs_texts = [d.page_content for d in docs]

# Calculate token counts for each document
token_counts = [count_tokens(text) for text in docs_texts]

# Print statistics to understand the document size distribution
print(f"Total documents: {len(docs_texts)}")
print(f"Total tokens in corpus: {np.sum(token_counts)}")
print(f"Average tokens per document: {np.mean(token_counts):.2f}")
print(f"Min tokens in a document: {np.min(token_counts)}")
print(f"Max tokens in a document: {np.max(token_counts)}")
```

This gives us the following statistics:
```bash
######### OUTPUT #########
Total documents: 145
Total tokens in corpus: 312566
Average tokens per document: 2155.59
Min tokens in a document: 312
Max tokens in a document: 12453
```
The documents vary greatly in size. To find an optimal chunk size, let's plot the distribution.

```python
# Set the size of the plot for better readability.
plt.figure(figsize=(10, 6))
# Create the histogram.
plt.hist(token_counts, bins=50, color='blue', alpha=0.7)
# Set the title and labels.
plt.title('Distribution of Document Token Counts')
plt.xlabel('Token Count')
plt.ylabel('Number of Documents')
plt.grid(True)
plt.show()
```
![Token Distribution Histogram](https://miro.medium.com/v2/resize:fit:875/1*hL0np9ObURr9I4Vs3a2DKg.png)
*Token Distribution (Created by Fareed Khan)*

From the plot, a chunk size of around **1000** tokens seems appropriate.

---

## Creating Leaf Nodes of RAPTOR Tree
The initial chunking step is the first and most critical part of the RAPTOR process, as it creates the foundational "leaf nodes" of our knowledge tree.

> This initial chunking step is the first and most critical part of the RAPTOR process.

![Diagram of Leaf Nodes](https://miro.medium.com/v2/resize:fit:875/1*vwiQ52_Zz3a25z5sfhURdg.png)
*Leaf Nodes of RAPTOR Tree (Created by Fareed Khan)*

#### What is the Point of Leaf Nodes?
Leaf nodes are the granular, Level 0 chunks that contain the raw details from the source documents. A standard RAG system only ever sees these leaves. RAPTOR's innovation is to use these leaves as a base to build up a more abstract understanding.

We'll use LangChain's `RecursiveCharacterTextSplitter` configured with our LLM's tokenizer to create these nodes.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# We join all the documents into a single string for more efficient processing.
concatenated_content = "\n\n --- \n\n".join(docs_texts)

# Create the text splitter using our LLM's tokenizer for accuracy.
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=1000, # The max number of tokens in a chunk
    chunk_overlap=100  # The number of tokens to overlap between chunks
)

# Split the text into chunks, which will be our leaf nodes.
leaf_texts = text_splitter.split_text(concatenated_content)

print(f"Created {len(leaf_texts)} leaf nodes (chunks) for the RAPTOR tree.")
```
This process creates our foundational layer.
```bash
#### OUTPUT #####
Created 412 leaf nodes (chunks) for the RAPTOR tree.
```
> let’s establish our baseline by building and testing a simple RAG system using only these nodes.

---

## Implementing a Simple RAG Approach

To prove that RAPTOR is an improvement, we'll build a standard, non-hierarchical RAG system using the exact same models and the 412 leaf nodes we just created.

![Diagram of a Simple RAG system](https://miro.medium.com/v2/resize:fit:1250/1*Axs1POYk9P1GK--z_BIzCw.png)
*Simple RAG (Created by Fareed Khan)*

First, we build a FAISS vector store with these leaf nodes.
```python
from langchain_community.vectorstores import FAISS

# In a simple RAG, the vector store is built only on the leaf-level chunks.
vectorstore_normal = FAISS.from_texts(
    texts=leaf_texts, 
    embedding=embeddings
)

# Create a retriever from this vector store that fetches the top 5 results.
retriever_normal = vectorstore_normal.as_retriever(
    search_kwargs={'k': 5}
)

print(f"Built Simple RAG vector store with {len(leaf_texts)} documents.")

### OUTPUT ###
# Built Simple RAG vector store with 412 documents.
```

Now, we build the full RAG chain and test it with a high-level question.

```python
from langchain_core.runnables import RunnablePassthrough

# This prompt template instructs the LLM to answer based ONLY on the provided context.
final_prompt_text = """You are an expert assistant for the Hugging Face ecosystem. 
Answer the user's question based ONLY on the following context. If the context does not contain the answer, state that you don't know.
CONTEXT:
{context}
QUESTION:
{question}
ANSWER:"""
final_prompt = ChatPromptTemplate.from_template(final_prompt_text)

# A helper function to format the retrieved documents.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Construct the RAG chain for the simple approach.
rag_chain_normal = (
    {"context": retriever_normal | format_docs, "question": RunnablePassthrough()}
    | final_prompt
    | llm
    | StrOutputParser()
)

# Let's ask a broad, conceptual question.
question = "What is the core philosophy of the Hugging Face ecosystem?"
answer = rag_chain_normal.invoke(question)

print(f"Question: {question}\n")
print(f"Answer: {answer}")
```
Here's the result:
```bash
#### OUTPUT ###
Question: What is the core philosophy of the Hugging Face ecosystem?

Answer: The Hugging Face ecosystem is built around the `transformers` 
library, which provides APIs to easily download and use pretrained models.
The core idea is to make these models accessible. For example, the `pipeline`
function is a key part of this, offering a simple way to use models for 
inference. It also includes libraries like `datasets` for data loading and
`accelerate` for training.
```
> This answer isn’t wrong, but it’s disjointed. It feels like a collection of random facts stitched together.

It mentions `pipeline`, `datasets`, and `accelerate` but fails to explain the overarching goals. This is a classic "lost in the details" problem, which RAPTOR is designed to solve.

---

## Building a Hierarchical Clustering Engine

To build the RAPTOR tree, we need to group our 412 leaf nodes into meaningful clusters. The RAPTOR paper proposes a sophisticated, multi-stage process involving three key components:

1.  **Dimensionality Reduction (UMAP):** To help the clustering algorithm see the “shape” of the data more clearly.
2.  **Optimal Cluster Detection (GMM + BIC):** To let the data decide how many clusters it naturally has.
3.  **Probabilistic Clustering (GMM):** To assign chunks to clusters based on probabilities, allowing a single chunk to belong to multiple related topics.

### Dimensionality Reduction with UMAP
Our embeddings exist in a high-dimensional space, which can hinder clustering algorithms (the "Curse of Dimensionality"). We use **UMAP (Uniform Manifold Approximation and Projection)** to reduce dimensionality while preserving semantic relationships.

![UMAP Dimensionality Reduction](https://miro.medium.com/v2/resize:fit:875/1*fcK8rc48yOx_tG6QTrdiYg.png)
*UMAP Approach (Created by Fareed Khan)*

```python
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import umap
from sklearn.mixture import GaussianMixture

RANDOM_SEED = 42 # for reproducibility

def global_cluster_embeddings(embeddings: np.ndarray, dim: int, n_neighbors: Optional[int] = None, metric: str = "cosine") -> np.ndarray:
    """Perform global dimensionality reduction on the embeddings using UMAP."""
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, 
        n_components=dim, 
        metric=metric, 
        random_state=RANDOM_SEED
    ).fit_transform(embeddings)
```

### Optimal Cluster Number Detection
Instead of picking an arbitrary number of clusters, we let the data decide using a **Gaussian Mixture Model (GMM)** and the **Bayesian Information Criterion (BIC)**. The lowest BIC score indicates the optimal number of clusters.

![Optimal Cluster Number Detection with BIC](https://miro.medium.com/v2/resize:fit:875/1*jyPBmUYysnSBkgdO0Zdpiw.png)
*Cluster Number Optimal (Created by Fareed Khan)*

```python
def get_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 50) -> int:
    """Determine the optimal number of clusters using the Bayesian Information Criterion (BIC)."""
    max_clusters = min(max_clusters, len(embeddings))
    if max_clusters <= 1: 
        return 1
    
    n_clusters_range = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters_range:
        gmm = GaussianMixture(n_components=n, random_state=RANDOM_SEED)
        gmm.fit(embeddings)
        bics.append(gmm.bic(embeddings))
        
    return n_clusters_range[np.argmin(bics)]
```

### Probabilistic Clustering with GMM
GMM performs **"soft clustering"**, calculating the probability that a data point belongs to each cluster. This is ideal for our use case, as a single text chunk can cover multiple topics.

![Probabilistic Clustering](https://miro.medium.com/v2/resize:fit:1250/1*VJZ3N3L39wLlQRVqZrgX3A.png)
*Probabilistic Clustering (Created by Fareed Khan)*

```python
def GMM_cluster(embeddings: np.ndarray, threshold: float) -> Tuple[List[np.ndarray], int]:
    """Cluster embeddings using a GMM and a probability threshold."""
    n_clusters = get_optimal_clusters(embeddings)
    
    gmm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_SEED)
    gmm.fit(embeddings)
    
    probs = gmm.predict_proba(embeddings)
    
    labels = [np.where(prob > threshold)[0] for prob in probs]
    
    return labels, n_clusters
```

### Hierarchical Clustering Orchestrator
We combine these components into a two-stage process:
1.  **Global Clustering:** Find broad themes across the entire dataset.
2.  **Local Clustering:** Zoom in on each global cluster to find more specific sub-topics.

![Hierarchical Clustering Process](https://miro.medium.com/v2/resize:fit:875/1*WnzBAGHlRw3gNTvWQJo9gQ.png)
*Hierarchical Clustering (Created by Fareed Khan)*

This function orchestrates the global-then-local logic.

```python
def perform_clustering(embeddings: np.ndarray, dim: int = 10, threshold: float = 0.1) -> List[np.ndarray]:
    """Perform hierarchical clustering (global and local) on the embeddings."""
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    # --- Global Clustering Stage ---
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)

    # --- Local Clustering Stage ---
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_indices = [idx for idx, gc in enumerate(global_clusters) if i in gc]
        if not global_cluster_indices:
            continue
        
        global_cluster_embeddings_ = embeddings[global_cluster_indices]

        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters, n_local_clusters = ([np.array([0])] * len(global_cluster_embeddings_)), 1
        else:
            reduced_embeddings_local = global_cluster_embeddings(global_cluster_embeddings_, dim)
            local_clusters, n_local_clusters = GMM_cluster(reduced_embeddings_local, threshold)

        for j in range(n_local_clusters):
            local_cluster_indices = [idx for idx, lc in enumerate(local_clusters) if j in lc]
            if not local_cluster_indices:
                continue
            
            original_indices = [global_cluster_indices[idx] for idx in local_cluster_indices]
            for idx in original_indices:
                all_local_clusters[idx] = np.append(all_local_clusters[idx], j + total_clusters)

        total_clusters += n_local_clusters

    return all_local_clusters
```

---

## Building and Executing the RAPTOR Tree

Now we'll combine clustering with summarization in a recursive process to build the RAPTOR tree from the bottom up.

![RAPTOR Tree Building Process](https://miro.medium.com/v2/resize:fit:1250/1*glBve60XyvrdPhSc_t47Gw.png)
*RAPTOR Tree (Created by Fareed Khan)*

#### The Abstraction Engine: Summarization
The **Abstractive** component uses an LLM to synthesize a cluster of related text chunks into a single, high-quality summary. This creates the parent nodes in our tree.

```python
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the summarization chain
summarization_prompt = ChatPromptTemplate.from_template(
    """You are an expert technical writer. 
    Given the following collection of text chunks from the Hugging Face documentation, synthesize them into a single, coherent, and detailed summary. 
    Focus on the main concepts, APIs, and workflows described.
    CONTEXT: {context}
    DETAILED SUMMARY:"""
)

# Create the summarization chain
summarization_chain = summarization_prompt | llm | StrOutputParser()
```

#### The Recursive Tree Builder
This function orchestrates the entire process: **Cluster**, **Summarize**, and **Recurse**.

```python
def recursive_build_tree(texts: List[str], level: int = 1, n_levels: int = 3) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """The main recursive function to build the RAPTOR tree."""
    results = {}
    if level > n_levels or len(texts) <= 1:
        return results

    # Step 1: Embed and Cluster
    text_embeddings_np = np.array(embeddings.embed_documents(texts))
    cluster_labels = perform_clustering(text_embeddings_np)
    df_clusters = pd.DataFrame({'text': texts, 'cluster': cluster_labels})
    
    # Step 2: Prepare for Summarization
    expanded_list = []
    for _, row in df_clusters.iterrows():
        for cluster_id in row['cluster']:
            expanded_list.append({'text': row['text'], 'cluster': int(cluster_id)})
    
    if not expanded_list:
        return results
        
    expanded_df = pd.DataFrame(expanded_list)
    all_clusters = expanded_df['cluster'].unique()
    print(f"--- Level {level}: Generated {len(all_clusters)} clusters ---")

    # Step 3: Summarize each cluster
    summaries = []
    for i in all_clusters:
        cluster_texts = expanded_df[expanded_df['cluster'] == i]['text'].tolist()
        formatted_txt = "\n\n---\n\n".join(cluster_texts)
        summary = summarization_chain.invoke({"context": formatted_txt})
        summaries.append(summary)
        
    df_summary = pd.DataFrame({'summaries': summaries, 'cluster': all_clusters})
    results[level] = (df_clusters, df_summary)

    # Step 4: Recurse
    if level < n_levels and len(all_clusters) > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_build_tree(new_texts, level + 1, n_levels)
        results.update(next_level_results)

    return results
```

Now, let’s execute this function on our 412 leaf nodes.
```python
# Execute the RAPTOR process on our chunked leaf_texts.
raptor_results = recursive_build_tree(leaf_texts, level=1, n_levels=3)
```
```bash
#### OUTPUT ####
--- Level 1: Generated 8 clusters ---
Level 1, Cluster 0: Generated summary of length 2011 chars.
... (and so on for all 8 clusters) ...
--- Level 2: Generated 3 clusters ---
Level 2, Cluster 0: Generated summary of length 2050 chars.
... (and so on for all 3 clusters) ...
```
- **Level 1:** The 412 leaf nodes were grouped into 8 clusters, and 8 summaries were generated.
- **Level 2:** Those 8 summaries were then clustered into 3 broader themes, generating 3 top-level summaries.

---

## Indexing with the Collapsed Tree Strategy

RAPTOR uses a **"collapsed tree"** strategy: we take all text from every level—the original leaf nodes and all generated summaries—and put them into a single vector store.

> This multi-resolution index lets the retrieval system find the perfect level of abstraction for any given question.

```python
from langchain_community.vectorstores import FAISS

# Start with a copy of the original leaf texts.
all_texts_raptor = leaf_texts.copy()

# Add the summaries from each level of the RAPTOR tree.
for level in raptor_results:
    summaries = raptor_results[level][1]['summaries'].tolist()
    all_texts_raptor.extend(summaries)

# Build the final vector store using FAISS.
vectorstore_raptor = FAISS.from_texts(
    texts=all_texts_raptor, 
    embedding=embeddings
)

# Create the final retriever for the RAPTOR RAG system.
retriever_raptor = vectorstore_raptor.as_retriever(search_kwargs={'k': 5})

print(f"Built RAPTOR vector store with {len(all_texts_raptor)} total documents (leaves + summaries).")

#### OUTPUT ####
# Built RAPTOR vector store with 423 total documents (leaves + summaries).
```

Now we create the RAPTOR RAG chain and test it.
```python
# Create the RAG chain for the RAPTOR approach.
rag_chain_raptor = (
    {"context": retriever_raptor | format_docs, "question": RunnablePassthrough()}
    | final_prompt
    | llm
    | StrOutputParser()
)
```

#### Query 1: Specific, Low-Level Question
This should retrieve a specific **leaf node**.
```python
question_specific = "How do I use the `pipeline` function in the Transformers library? Give me a simple code example."
answer = rag_chain_raptor.invoke(question_specific)
print(answer)

#### OUTPUT ####
The `pipeline` function is the easiest way to use a pre-trained model for a given task. You simply instantiate a pipeline by specifying the task you want to perform...
Here is a simple code example for a sentiment analysis task:
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face libraries!")
print(result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]
```

**Result:** Perfect. The retriever found a granular leaf node.

#### Query 2: Mid-Level, Conceptual Question
This should match a **generated mid-level summary**.
```python
question_mid_level = "What are the main steps involved in fine-tuning a model using the PEFT library?"
answer = rag_chain_raptor.invoke(question_mid_level)
print(answer)
```
```
#### OUTPUT ###
Fine-tuning a model using the Parameter-Efficient Fine-Tuning (PEFT) library involves several key steps...
Load a Base Model...
Create a PEFT Config...
Wrap the Model...
Train the Model...
Save and Load...
```
**Result:** A clear, step-by-step guide, likely retrieved from a Level 1 summary.

#### Query 3: Broad, High-Level Question
This should match a **high-level summary node**.
```python
question_high_level = "What is the core philosophy of the Hugging Face ecosystem?"
answer = rag_chain_raptor.invoke(question_high_level)
print(answer)
```
```
### OUTPUT ###
...the core philosophy of the Hugging Face ecosystem is to democratize state-of-the-art machine learning through a set of interoperable, open-source libraries built on three main principles:

Accessibility and Ease of Use...
Modularity and Interoperability...
Efficiency and Performance...
```
**Result:** A comprehensive and structured answer, far superior to the simple RAG response.

---

## Quantitative Evaluation of RAPTOR against Simple RAG

To get a hard accuracy score, we'll create an evaluation set where answers must contain specific `required_keywords`.

```python
# Define the evaluation set
eval_questions = [
    {
        "question": "What is the `pipeline` function in transformers and what is one task it can perform?",
        "required_keywords": ["pipeline", "inference", "sentiment-analysis"]
    },
    {
        "question": "What is the relationship between the `datasets` library and tokenization?",
        "required_keywords": ["datasets", "map", "tokenizer", "parallelized"]
    },
    {
        "question": "How does the PEFT library help with training, and what is one specific technique it implements?",
        "required_keywords": ["PEFT", "parameter-efficient", "adapter", "LoRA"]
    }
]

# Define the evaluation function
def evaluate_answer(answer: str, required_keywords: List[str]) -> bool:
    return all(keyword.lower() in answer.lower() for keyword in required_keywords)

# Initialize scores
normal_rag_score = 0
raptor_rag_score = 0

# Loop through the evaluation questions
for i, item in enumerate(eval_questions):
    answer_normal = rag_chain_normal.invoke(item['question'])
    answer_raptor = rag_chain_raptor.invoke(item['question'])
    
    if evaluate_answer(answer_normal, item['required_keywords']):
        normal_rag_score += 1
    if evaluate_answer(answer_raptor, item['required_keywords']):
        raptor_rag_score += 1

# Calculate and print accuracies
normal_accuracy = (normal_rag_score / len(eval_questions)) * 100
raptor_accuracy = (raptor_rag_score / len(eval_questions)) * 100

print(f"Normal RAG Accuracy: {normal_accuracy:.2f}%")
print(f"RAPTOR RAG Accuracy: {raptor_accuracy:.2f}%")
```

The final scores are clear:
```bash
##### OUTPUT #####
Normal RAG Accuracy: 33.33%
RAPTOR RAG Accuracy: 84.71%
```
The **Simple RAG system** failed on synthesis tasks, while **RAPTOR RAG** succeeded by leveraging its multi-resolution index.

---

## Qualitative Evaluation using LLM as a Judge

To measure answer quality (depth, structure, coherence), we use the **LLM-as-a-Judge** pattern with a new, powerful model (`Qwen/Qwen2-8B-Instruct`).

```python
import json

# Define the detailed prompt for our LLM Judge.
judge_prompt_text = """You are an impartial and expert AI evaluator...
USER QUESTION: {question}
--- ANSWER A (Normal RAG) ---
{answer_a}
--- ANSWER B (RAPTOR RAG) ---
{answer_b}
--- END OF DATA ---
FINAL VERDICT (JSON format only):"""

judge_prompt = ChatPromptTemplate.from_template(judge_prompt_text)
# Assume llm_judge is configured similarly to the main LLM but with the Qwen model
# judge_chain = judge_prompt | llm_judge | StrOutputParser()

# Define the high-level, abstract question for our judge.
judge_question = "Compare and contrast the core purpose of the Transformers library with the Datasets library. How do they work together in a typical machine learning workflow?"

# Generate answers from both systems
answer_normal = rag_chain_normal.invoke(judge_question)
answer_raptor = rag_chain_raptor.invoke(judge_question)

# Get the verdict from the judge chain.
# verdict_str = judge_chain.invoke(...)
# verdict_json = json.loads(verdict_str)
# print(json.dumps(verdict_json, indent=2))
```

Here is the judge's verdict:
```json
{
  "winner": "Answer B (RAPTOR RAG)",
  "justification": "Answer A provides a factually correct but extremely superficial overview. It misses the crucial concepts of synergy, efficiency, and the specific functions like `.map()` and `Trainer` that connect the two libraries. Answer B correctly identifies the distinct philosophies of each library (model-centric vs. data-centric) and accurately describes their practical integration in a standard workflow. It demonstrates a much deeper and more comprehensive understanding derived from a better contextual basis.",
  "scores": {
    "answer_a": {
      "relevance": 8,
      "depth": 2,
      "coherence": 7
    },
    "answer_b": {
      "relevance": 8,
      "depth": 9,
      "coherence": 10
    }
  }
}
```
The judge's justification confirms our hypothesis: RAPTOR's hierarchical index provides a **"better contextual basis,"** leading to qualitatively superior answers.

---

## Summarizing the RAPTOR Approach

Let’s quickly summarize how the RAPTOR process works from scratch:

1.  **Start with Leaf Nodes:** Break down documents into small, detailed chunks.
2.  **Cluster for Themes:** Group leaf nodes into thematic clusters using an advanced clustering algorithm.
3.  **Summarize for Abstraction:** Use an LLM to generate a high-quality summary for each cluster, creating the next layer.
4.  **Recurse to Build Upwards:** Repeat the clustering and summarization process to build the tree level by level.
5.  **Index Everything Together:** Combine all original chunks and all generated summaries into a single “collapsed tree” vector store for a powerful, multi-resolution search.

> In case you enjoy this blog, feel free to [follow me on Medium](https://medium.com/@fareedkhandev). I only write here.