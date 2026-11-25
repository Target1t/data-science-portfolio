import os
import json

import faiss
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


API_KEY = os.getenv(
    "DASHSCOPE_API_KEY"
)

WORKSPACE_ID = os.getenv(
    "DASHSCOPE_WORKSPACE_ID"
)


client = OpenAI(

    api_key=API_KEY,

    base_url=(
        f"https://{WORKSPACE_ID}"
        ".cn-beijing.maas.aliyuncs.com/"
        "compatible-mode/v1"
    )

)



def embed_query(query):


    response = client.embeddings.create(

        model="qwen3.7-text-embedding",

        input=[query],

        encoding_format="float"

    )


    vector = np.array(

        response.data[0].embedding,

        dtype="float32"

    )


    faiss.normalize_L2(
        vector.reshape(1,-1)
    )


    return vector



def retrieve(

    doc_name,

    query,

    top_k=20

):


    index_path = (

        f"data/index/"
        f"{doc_name}.faiss"

    )


    metadata_path = (

        f"data/index/"
        f"{doc_name}_metadata.json"

    )


    if not os.path.exists(index_path):

        raise FileNotFoundError(
            index_path
        )


    index = faiss.read_index(
        index_path
    )


    with open(

        metadata_path,

        encoding="utf-8"

    ) as f:

        chunks=json.load(f)



    query_vector = embed_query(
        query
    )



    scores, ids = index.search(

        query_vector.reshape(1,-1),

        top_k

    )



    results=[]



    for score, idx in zip(

        scores[0],

        ids[0]

    ):


        chunk = chunks[idx]


        results.append({

            "text":
            chunk["text"],


            "metadata":
            chunk["metadata"],


            "retrieval_score":
            float(score)

        })


    return results