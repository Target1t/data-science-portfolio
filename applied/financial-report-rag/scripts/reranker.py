from FlagEmbedding import FlagReranker


# ============================================================
# Load reranker once
# ============================================================

reranker = FlagReranker(
    "BAAI/bge-reranker-v2-m3",
    use_fp16=True
)


# ============================================================
# Build cleaner text for reranking
# ============================================================

def build_rerank_text(document):

    """
    Construct reranker input.

    We keep meaningful section information,
    but avoid feeding useless metadata such as:
    - internal document id
    - retrieval score
    - duplicated page labels
    """

    metadata = document.get("metadata", {})

    section = metadata.get(
        "section",
        "Unknown"
    )

    text = document.get(
        "text",
        ""
    )


    # Remove our manually added metadata prefix if present.
    # The real PDF content normally begins around Markdown headings.
    heading_positions = []

    for marker in [
        "######",
        "#####",
        "####",
        "###",
        "## "
    ]:

        pos = text.find(marker)

        if pos != -1:
            heading_positions.append(pos)


    if heading_positions:

        start = min(heading_positions)

        body = text[start:]

    else:

        body = text


    rerank_text = f"""
Section:
{section}

Content:
{body}
""".strip()


    return rerank_text


# ============================================================
# Rerank
# ============================================================

def rerank(
    query,
    documents
):

    if not documents:
        return []


    pairs = []


    for doc in documents:

        rerank_text = build_rerank_text(
            doc
        )

        pairs.append(
            [
                query,
                rerank_text
            ]
        )


    scores = reranker.compute_score(
        pairs
    )


    # One-document case may return float
    if isinstance(
        scores,
        (int, float)
    ):

        scores = [scores]


    ranked_results = []


    for score, doc in zip(
        scores,
        documents
    ):

        # copy to avoid unexpected mutation
        item = doc.copy()

        item["rerank_score"] = float(
            score
        )

        ranked_results.append(
            item
        )


    ranked_results.sort(

        key=lambda x: x["rerank_score"],

        reverse=True

    )


    return ranked_results