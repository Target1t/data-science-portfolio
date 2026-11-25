import json
from pathlib import Path


QA_PATH = Path(
    "data/financebench/financebench_open_source.jsonl"
)

PDF_DIR = Path("data/raw")


# ============================================================
# Load JSONL
# ============================================================

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

        item = json.loads(line)

        questions.append(item)


# ============================================================
# Collect unique document names
# ============================================================

doc_names = sorted(
    set(
        item["doc_name"]
        for item in questions
    )
)


print("=" * 70)
print("FinanceBench document check")
print("=" * 70)

print(
    f"\nTotal QA: {len(questions)}"
)

print(
    f"Unique documents: {len(doc_names)}"
)


# ============================================================
# Check PDF files
# ============================================================

existing = []
missing = []


for i, doc_name in enumerate(
    doc_names,
    start=1
):

    pdf_path = PDF_DIR / f"{doc_name}.pdf"


    if pdf_path.exists():

        status = "OK"

        existing.append(
            doc_name
        )

    else:

        status = "MISSING"

        missing.append(
            doc_name
        )


    print(
        f"{i:02d}. "
        f"{doc_name}.pdf "
        f"[{status}]"
    )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

print(
    "Existing PDFs:",
    len(existing)
)

print(
    "Missing PDFs:",
    len(missing)
)


if missing:

    print(
        "\nPlease download these PDFs:"
    )

    for doc_name in missing:

        print(
            f"{doc_name}.pdf"
        )

else:

    print(
        "\nAll required FinanceBench PDFs are ready."
    )