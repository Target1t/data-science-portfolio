import json
import subprocess
import sys
from pathlib import Path


QA_PATH = Path(
    "data/financebench/financebench_open_source.jsonl"
)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("data/index")


def load_doc_names():
    doc_names = set()

    with open(QA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            item = json.loads(line)
            doc_names.add(item["doc_name"])

    return sorted(doc_names)


def run_script(script_name, doc_name):
    command = [
        sys.executable,
        f"scripts/{script_name}",
        doc_name,
    ]

    subprocess.run(
        command,
        check=True,
    )


def prepare_document(doc_name):
    print("\n" + "=" * 80)
    print(f"Document: {doc_name}")
    print("=" * 80)

    pdf_path = RAW_DIR / f"{doc_name}.pdf"

    pages_path = (
        PROCESSED_DIR /
        f"{doc_name}_pages.json"
    )

    chunks_path = (
        PROCESSED_DIR /
        f"{doc_name}_chunks.json"
    )

    index_path = (
        INDEX_DIR /
        f"{doc_name}.faiss"
    )

    metadata_path = (
        INDEX_DIR /
        f"{doc_name}_metadata.json"
    )


    # --------------------------------------------------------
    # Check PDF
    # --------------------------------------------------------

    if not pdf_path.exists():
        print(f"[ERROR] Missing PDF: {pdf_path}")
        return False


    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    if pages_path.exists():

        print(
            f"[SKIP] Parsed pages already exist: "
            f"{pages_path}"
        )

    else:

        print("[1/3] Parsing PDF...")

        run_script(
            "parse_pdf.py",
            doc_name
        )


    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    if chunks_path.exists():

        print(
            f"[SKIP] Chunks already exist: "
            f"{chunks_path}"
        )

    else:

        print("[2/3] Building chunks...")

        run_script(
            "build_chunks.py",
            doc_name
        )


    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    if (
        index_path.exists()
        and metadata_path.exists()
    ):

        print(
            f"[SKIP] Index already exists: "
            f"{index_path}"
        )

    else:

        print("[3/3] Building FAISS index...")

        run_script(
            "build_index.py",
            doc_name
        )


    print(f"[DONE] {doc_name}")

    return True


def main():

    doc_names = load_doc_names()

    print("=" * 80)
    print("FinanceBench Offline Preparation")
    print("=" * 80)

    print(f"Unique documents: {len(doc_names)}")


    # --------------------------------------------------------
    # Smoke test
    # --------------------------------------------------------
    #
    # 先只处理前三份文档。
    # 验证无误后，把 [:3] 改成 doc_names。
    #

    selected_docs = doc_names


    print(
        f"Documents selected this run: "
        f"{len(selected_docs)}"
    )


    success = []
    failed = []


    for i, doc_name in enumerate(
        selected_docs,
        start=1
    ):

        print(
            f"\n[{i}/{len(selected_docs)}]"
        )

        try:

            ok = prepare_document(
                doc_name
            )

            if ok:
                success.append(doc_name)
            else:
                failed.append(doc_name)

        except Exception as e:

            print(
                f"\n[FAILED] {doc_name}"
            )

            print(
                f"Reason: {e}"
            )

            failed.append(doc_name)


    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print(
        "Successful:",
        len(success)
    )

    print(
        "Failed:",
        len(failed)
    )


    if failed:

        print("\nFailed documents:")

        for doc_name in failed:
            print(doc_name)


if __name__ == "__main__":
    main()