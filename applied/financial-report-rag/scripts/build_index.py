import json
import sys
from pathlib import Path

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# ============================================================
# Local embedding model
# ============================================================

EMBEDDING_MODEL = (
    "BAAI/bge-base-en-v1.5"
)


print(
    "Loading embedding model:",
    EMBEDDING_MODEL
)


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


print(
    "Embedding model loaded"
)



# ============================================================
# Paths
# ============================================================

def get_paths(doc_name):

    chunk_path = Path(
        f"data/processed/{doc_name}_chunks.json"
    )


    index_dir = Path(
        "data/index"
    )


    index_path = (
        index_dir /
        f"{doc_name}.faiss"
    )


    metadata_path = (
        index_dir /
        f"{doc_name}_metadata.json"
    )


    return (
        chunk_path,
        index_path,
        metadata_path
    )



# ============================================================
# Embedding
# ============================================================

def encode_texts(texts):


    embeddings = embedding_model.encode(

        texts,

        batch_size=16,

        normalize_embeddings=True,

        show_progress_bar=True

    )


    return embeddings



# ============================================================
# Main
# ============================================================

def main():


    if len(sys.argv) < 2:

        print(
            "Usage:"
        )

        print(
            "python scripts/build_index.py <document>"
        )

        return



    doc_name = sys.argv[1]


    (
        chunk_path,
        index_path,
        metadata_path

    ) = get_paths(
        doc_name
    )



    print("=" * 60)

    print(
        "Document:",
        doc_name
    )

    print(
        "Loading:",
        chunk_path
    )

    print("=" * 60)



    with open(

        chunk_path,

        encoding="utf-8"

    ) as f:


        chunks = json.load(f)



    print(
        "Original chunks:",
        len(chunks)
    )



    # ========================================================
    # Filter invalid chunks
    # ========================================================


    valid_chunks=[]


    for chunk in chunks:


        text = chunk.get(
            "text",
            ""
        )


        if not isinstance(
            text,
            str
        ):

            continue



        text=text.strip()



        if len(text)<30:

            continue



        valid_chunks.append(
            chunk
        )



    chunks=valid_chunks



    print(
        "Valid chunks:",
        len(chunks)
    )



    texts=[

        c["text"]

        for c in chunks

    ]



    # ========================================================
    # Embedding
    # ========================================================


    print(
        "Generating embeddings..."
    )



    vectors = encode_texts(
        texts
    )



    print(
        "Embedding shape:",
        vectors.shape
    )



    # ========================================================
    # Check
    # ========================================================


    if len(vectors)!=len(chunks):

        raise RuntimeError(

            "Embedding count does not match chunks"

        )



    dimension = vectors.shape[1]


    print(
        "Vector dimension:",
        dimension
    )



    # ========================================================
    # FAISS
    # ========================================================


    index = faiss.IndexFlatIP(
        dimension
    )


    index.add(
        vectors.astype(
            "float32"
        )
    )



    print(
        "FAISS vectors:",
        index.ntotal
    )



    # ========================================================
    # Save
    # ========================================================


    index_path.parent.mkdir(
        exist_ok=True
    )



    faiss.write_index(

        index,

        str(index_path)

    )



    with open(

        metadata_path,

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            chunks,

            f,

            ensure_ascii=False,

            indent=2

        )



    print("=" * 60)

    print(
        "Saved FAISS:"
    )

    print(
        index_path
    )


    print(
        "Saved metadata:"
    )

    print(
        metadata_path
    )

    print("=" * 60)



if __name__ == "__main__":

    main()