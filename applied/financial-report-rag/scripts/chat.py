import os

from dotenv import load_dotenv
from openai import OpenAI

from pipeline import search


# ============================================================
# Environment
# ============================================================

load_dotenv()

API_KEY = os.getenv("DASHSCOPE_API_KEY")
WORKSPACE_ID = os.getenv("DASHSCOPE_WORKSPACE_ID")

if not API_KEY:
    raise ValueError("DASHSCOPE_API_KEY is missing in .env")

if not WORKSPACE_ID:
    raise ValueError("DASHSCOPE_WORKSPACE_ID is missing in .env")


BASE_URL = (
    f"https://{WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/"
    "compatible-mode/v1"
)


client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


GENERATION_MODEL = "qwen3.7-plus"


# ============================================================
# Build context
# ============================================================

def build_context(results):
    """
    Convert retrieved chunks into LLM context.

    Keep explicit source metadata so the model can
    associate statements with pages and sections.
    """

    context_parts = []


    for rank, result in enumerate(results, start=1):

        metadata = result.get("metadata", {})

        page = metadata.get("page", "Unknown")
        section = metadata.get("section", "Unknown")
        chunk_type = metadata.get("type", "Unknown")

        text = result.get("text", "")


        context_part = f"""
[Evidence {rank}]
Page: {page}
Section: {section}
Type: {chunk_type}

{text}
""".strip()


        context_parts.append(context_part)


    return "\n\n".join(context_parts)


# ============================================================
# Generate answer
# ============================================================

def generate_answer(
    doc_name,
    question,
    retrieve_k=30,
    final_k=3
):

    # --------------------------------------------------------
    # Step 1: Retrieval + reranking
    # --------------------------------------------------------

    results = search(
        doc_name=doc_name,
        query=question,
        retrieve_k=retrieve_k,
        final_k=final_k,
        use_query_enhancement=True
    )


    # --------------------------------------------------------
    # Step 2: Build evidence context
    # --------------------------------------------------------

    context = build_context(results)


    # --------------------------------------------------------
    # Step 3: Prompt
    # --------------------------------------------------------

    system_prompt = """
You are a financial report question-answering assistant.

Answer the user's question using ONLY the evidence provided.

Rules:

1. Do not use outside knowledge.
2. Do not invent financial figures.
3. Preserve the units shown in the evidence.
4. If the evidence is insufficient, explicitly say:
   "The provided evidence is insufficient to answer the question."
5. Give a concise answer.
6. Cite the supporting page number after the answer.
7. Prefer the evidence that most directly answers the question.
8. Do not claim a page supports the answer unless that page actually contains
   the relevant information.
""".strip()


    user_prompt = f"""
Question:
{question}

Retrieved evidence:

{context}

Answer the question based only on the retrieved evidence.
""".strip()


    # --------------------------------------------------------
    # Step 4: Qwen generation
    # --------------------------------------------------------

    response = client.chat.completions.create(

        model=GENERATION_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.1
    )


    answer = response.choices[0].message.content


    return {
        "answer": answer,
        "sources": results
    }


# ============================================================
# CLI test
# ============================================================

if __name__ == "__main__":

    DOC_NAME = "apple_2024_10k"

    QUESTION = (
        "What was Apple's net income "
        "in fiscal year 2024?"
    )


    result = generate_answer(
        doc_name=DOC_NAME,
        question=QUESTION
    )


    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)

    print(QUESTION)


    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])


    print("\n" + "=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)


    for rank, source in enumerate(
        result["sources"],
        start=1
    ):

        metadata = source.get(
            "metadata",
            {}
        )


        print(
            f"\nRank {rank}"
        )

        print(
            "Page:",
            metadata.get("page")
        )

        print(
            "Section:",
            metadata.get("section")
        )

        print(
            "FAISS score:",
            round(
                source.get(
                    "retrieval_score",
                    0
                ),
                4
            )
        )

        print(
            "Rerank score:",
            round(
                source.get(
                    "rerank_score",
                    0
                ),
                4
            )
        )