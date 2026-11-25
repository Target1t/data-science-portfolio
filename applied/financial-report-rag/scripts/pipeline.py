import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker


# ======================================================
# Models
# ======================================================

embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)


reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=True
)



# ======================================================
# Retrieve
# ======================================================

def dense_retrieve(
    query,
    index_path,
    metadata_path,
    top_k=20
):


    index = faiss.read_index(
        index_path
    )


    with open(
        metadata_path,
        encoding="utf-8"
    ) as f:

        metadata=json.load(f)



    query_vector = embedding_model.encode(

        [query],

        normalize_embeddings=True

    )


    scores, ids = index.search(

        np.array(
            query_vector,
            dtype="float32"
        ),

        top_k

    )



    results=[]


    for score,idx in zip(

        scores[0],

        ids[0]

    ):


        item=metadata[idx]


        results.append({

            "content":
                item["text"],


            "metadata":
                item["metadata"],


            "faiss_score":
                float(score)

        })



    return results




# ======================================================
# Rerank
# ======================================================

def rerank(
    query,
    documents,
    top_k=5
):


    pairs=[]


    for doc in documents:


        pairs.append(

            [
                query,
                doc["content"]
            ]

        )



    scores = reranker.compute_score(
        pairs
    )



    ranked = sorted(

        zip(
            scores,
            documents
        ),

        key=lambda x:x[0],

        reverse=True

    )



    results=[]


    for score,doc in ranked[:top_k]:


        doc["rerank_score"]=float(score)


        results.append(
            doc
        )


    return results




# ======================================================
# Full Pipeline
# ======================================================

def search(

    query,

    document,

    top_k=5

):


    index_path = (

        f"data/index/"
        f"{document}.faiss"

    )


    metadata_path = (

        f"data/index/"
        f"{document}_metadata.json"

    )



    # Stage 1
    candidates = dense_retrieve(

        query,

        index_path,

        metadata_path,

        top_k=20

    )


    # Stage 2

    final_results = rerank(

        query,

        candidates,

        top_k

    )


    return final_results




# ======================================================
# Test
# ======================================================

if __name__=="__main__":


    query=(

        "What is the FY2018 capital expenditure "
        "amount (in USD millions) for 3M? "
        "Answer based on the cash flow statement."

    )


    results = search(

        query,

        "3M_2018_10K"

    )



    print("="*70)

    print(
        "QUERY:"
    )

    print(
        query
    )


    print("="*70)



    for i,r in enumerate(results):


        print()

        print(
            "Rank:",
            i+1
        )


        print(
            "Page:",
            r["metadata"].get(
                "page"
            )
        )


        print(
            "FAISS:",
            r["faiss_score"]
        )


        print(
            "RERANK:",
            r["rerank_score"]
        )


        print()

        print(
            r["content"][:500]
        )


        print(
            "-"*70
        )