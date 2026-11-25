import sys
import json
import faiss

import numpy as np

from sentence_transformers import SentenceTransformer



# ==================================================
# Model
# ==================================================

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)



# ==================================================
# Retrieve
# ==================================================

def retrieve(
    doc_name,
    query,
    top_k=5
):


    index_path = (
        f"data/index/{doc_name}.faiss"
    )


    metadata_path = (
        f"data/index/{doc_name}_metadata.json"
    )


    index = faiss.read_index(
        index_path
    )


    with open(
        metadata_path,
        encoding="utf-8"
    ) as f:

        metadata=json.load(f)



    q_vector = model.encode(

        [
            query
        ],

        normalize_embeddings=True

    )



    scores, ids = index.search(

        np.array(
            q_vector,
            dtype="float32"
        ),

        top_k

    )



    results=[]


    for score, idx in zip(

        scores[0],

        ids[0]

    ):


        item=metadata[idx]


        item["score"]=float(score)


        results.append(
            item
        )


    return results




# ==================================================
# Main
# ==================================================

if __name__=="__main__":


    if len(sys.argv)<2:

        print(
            "Usage:"
        )

        print(
            "python test_retrieve.py <document>"
        )

        exit()



    doc=sys.argv[1]



    question=(

        "What is the FY2018 capital expenditure "
        "amount (in USD millions) for 3M? "
        "Give a response based on the cash flow statement."

    )



    print("="*60)

    print(
        "Document:",
        doc
    )


    print(
        "Question:",
        question
    )

    print("="*60)



    results=retrieve(

        doc,

        question,

        top_k=5

    )



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
            "Score:",
            r["score"]
        )


        print(
            r["text"][:500]
        )


        print("-"*60)