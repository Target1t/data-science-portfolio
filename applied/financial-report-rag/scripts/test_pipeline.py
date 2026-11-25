from pipeline import search


DOC_NAME = "apple_2024_10k"

QUESTION = (
    "What was Apple's net income "
    "in fiscal year 2024?"
)


results = search(

    DOC_NAME,

    QUESTION,

    retrieve_k=30,

    final_k=3,

    use_query_enhancement=True

)


print(
    "\nQuestion:",
    QUESTION
)


print(
    "\nDocument:",
    DOC_NAME
)


print(
    "\n========== Final Top Results =========="
)


for rank, result in enumerate(
    results,
    start=1
):


    metadata = result[
        "metadata"
    ]


    print("\n" + "=" * 70)


    print(
        "Rank:",
        rank
    )


    print(
        "Page:",
        metadata.get(
            "page"
        )
    )


    print(
        "Type:",
        metadata.get(
            "type"
        )
    )


    print(
        "Section:",
        metadata.get(
            "section"
        )
    )


    print(
        "FAISS score:",
        round(
            result.get(
                "retrieval_score",
                0
            ),
            4
        )
    )


    print(
        "Rerank score:",
        round(
            result.get(
                "rerank_score",
                0
            ),
            4
        )
    )


    print(
        "\nContent preview:\n"
    )


    print(
        result["text"][:800]
    )