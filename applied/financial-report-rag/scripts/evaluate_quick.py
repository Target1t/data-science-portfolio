import json
import re
from pathlib import Path

from retrieve import retrieve
from reranker import rerank


# ============================================================
# Config
# ============================================================

QA_PATH = Path(
    "data/financebench/financebench_open_source.jsonl"
)

PROCESSED_DIR = Path(
    "data/processed"
)

INDEX_DIR = Path(
    "data/index"
)


# 先测试少量
MAX_QUESTIONS = 2


# Dense retrieval recall 优先
RETRIEVE_K = 100



# ============================================================
# Text normalize
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = text.lower()


    text = re.sub(
        r"[^a-z0-9$%.\-]+",
        " ",
        text
    )


    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()



# ============================================================
# Load FinanceBench
# ============================================================

def load_questions():

    questions = []


    with open(
        QA_PATH,
        "r",
        encoding="utf-8"
    ) as f:


        for line in f:

            line = line.strip()


            if not line:
                continue


            questions.append(
                json.loads(line)
            )


    return questions



# ============================================================
# Check index
# ============================================================

def index_exists(doc_name):

    return (

        (
            INDEX_DIR /
            f"{doc_name}.faiss"
        ).exists()

        and

        (
            INDEX_DIR /
            f"{doc_name}_metadata.json"
        ).exists()

        and

        (
            PROCESSED_DIR /
            f"{doc_name}_pages.json"
        ).exists()

    )



# ============================================================
# Page cache
# ============================================================

PAGE_CACHE = {}



def load_pages(doc_name):

    if doc_name in PAGE_CACHE:

        return PAGE_CACHE[doc_name]


    path = (
        PROCESSED_DIR /
        f"{doc_name}_pages.json"
    )


    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        pages = json.load(f)


    PAGE_CACHE[doc_name] = pages


    return pages



# ============================================================
# FinanceBench evidence page
# -> our PDF physical page
# ============================================================

def locate_evidence_pages(item):


    doc_name = item["doc_name"]


    pages = load_pages(
        doc_name
    )


    normalized_pages = []


    for page in pages:


        page_text = normalize_text(
            page.get(
                "text",
                ""
            )
        )


        page_number = (
            page
            .get("metadata", {})
            .get("page_number")
        )


        if page_number is None:

            continue


        normalized_pages.append(
            (
                int(page_number),
                page_text
            )
        )



    matched = set()



    for evidence in item.get(
        "evidence",
        []
    ):


        evidence_text = normalize_text(
            evidence.get(
                "evidence_text",
                ""
            )
        )


        if not evidence_text:

            continue



        # ----------------------------
        # exact match
        # ----------------------------

        for page_number, page_text in normalized_pages:


            if evidence_text in page_text:


                matched.add(
                    page_number
                )



        if matched:

            continue



        # ----------------------------
        # token overlap
        # ----------------------------

        evidence_tokens = set(
            evidence_text.split()
        )


        best_page = None

        best_score = 0



        for page_number, page_text in normalized_pages:


            page_tokens = set(
                page_text.split()
            )


            overlap = len(
                evidence_tokens
                &
                page_tokens
            )


            if len(evidence_tokens) == 0:

                continue


            score = (
                overlap
                /
                len(evidence_tokens)
            )


            if score > best_score:

                best_score = score

                best_page = page_number



        if (

            best_page is not None

            and

            best_score >= 0.55

        ):

            matched.add(
                best_page
            )



    return matched



# ============================================================
# Find rank
# ============================================================

def first_relevant_rank(
    results,
    gold_pages
):


    for rank, result in enumerate(
        results,
        start=1
    ):


        page = (
            result
            .get("metadata", {})
            .get("page")
        )


        if page is None:

            continue


        try:

            page = int(page)


        except:

            continue



        if page in gold_pages:

            return rank



    return None



# ============================================================
# Metrics
# ============================================================

def calc_metric(ranks):


    total = len(ranks)


    if total == 0:

        return {
            "Recall@1":0,
            "Recall@3":0,
            "Recall@10":0,
            "MRR@10":0
        }



    recall1 = sum(

        r is not None
        and r <= 1

        for r in ranks

    ) / total



    recall3 = sum(

        r is not None
        and r <= 3

        for r in ranks

    ) / total



    recall10 = sum(

        r is not None
        and r <= 10

        for r in ranks

    ) / total



    mrr = sum(

        1/r

        for r in ranks

        if r is not None
        and r <=10

    ) / total



    return {

        "Recall@1": recall1,

        "Recall@3": recall3,

        "Recall@10": recall10,

        "MRR@10": mrr

    }



# ============================================================
# Main
# ============================================================

def main():


    questions = load_questions()


    selected = []



    for item in questions:


        doc_name = item["doc_name"]



        if not index_exists(
            doc_name
        ):

            continue



        gold_pages = locate_evidence_pages(
            item
        )



        if not gold_pages:

            continue



        selected.append(
            (
                item,
                gold_pages
            )
        )



    selected = selected[
        :MAX_QUESTIONS
    ]



    print("="*70)

    print(
        "FinanceBench Retrieval Evaluation"
    )

    print("="*70)


    print(
        "Eligible questions:",
        len(selected)
    )



    faiss_ranks = []

    bge_ranks = []



    for idx, (
        item,
        gold_pages
    ) in enumerate(
        selected,
        start=1
    ):


        doc_name = item["doc_name"]

        question = item["question"]



        print(
            "\n"
            +
            "-"*70
        )


        print(
            f"[{idx}/{len(selected)}]"
        )


        print(
            "Document:",
            doc_name
        )


        print(
            "Question:",
            question
        )


        print(
            "Mapped PDF page:",
            sorted(gold_pages)
        )



        # ==========================
        # FAISS
        # ==========================

        candidates = retrieve(

            doc_name,

            question,

            top_k=RETRIEVE_K

        )



        print(

            "\nFAISS Top10 pages:",

            [

                x["metadata"]["page"]

                for x in candidates[:10]

            ]

        )



        faiss_rank = first_relevant_rank(

            candidates,

            gold_pages

        )



        # ==========================
        # BGE
        # ==========================

        ranked = rerank(

            question,

            candidates

        )



        print(

            "BGE Top10 pages:",

            [

                x["metadata"]["page"]

                for x in ranked[:10]

            ]

        )



        bge_rank = first_relevant_rank(

            ranked,

            gold_pages

        )



        print(
            "FAISS gold rank:",
            faiss_rank
        )


        print(
            "BGE gold rank:",
            bge_rank
        )



        faiss_ranks.append(
            faiss_rank
        )


        bge_ranks.append(
            bge_rank
        )



    print(
        "\n"
        +
        "="*70
    )

    print(
        "RESULTS"
    )

    print(
        "="*70
    )



    print(
        "\nFAISS-only"
    )

    print(
        calc_metric(
            faiss_ranks
        )
    )


    print(
        "\nFAISS+BGE"
    )

    print(
        calc_metric(
            bge_ranks
        )
    )



if __name__ == "__main__":

    main()