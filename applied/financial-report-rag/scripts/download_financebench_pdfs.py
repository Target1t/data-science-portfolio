import json
import time
from pathlib import Path
from urllib.parse import quote

import requests


QA_PATH = Path(
    "data/financebench/financebench_open_source.jsonl"
)

OUTPUT_DIR = Path("data/raw")

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "patronus-ai/financebench/main/pdfs"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load FinanceBench QA
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

        questions.append(
            json.loads(line)
        )


# ============================================================
# Get unique required documents
# ============================================================

doc_names = sorted(
    {
        item["doc_name"]
        for item in questions
    }
)


print("=" * 70)
print("FinanceBench PDF Downloader")
print("=" * 70)

print(
    f"QA count: {len(questions)}"
)

print(
    f"Unique documents: {len(doc_names)}"
)


# ============================================================
# Download
# ============================================================

success = []
failed = []
skipped = []


for index, doc_name in enumerate(
    doc_names,
    start=1
):

    filename = f"{doc_name}.pdf"

    output_path = (
        OUTPUT_DIR / filename
    )


    # Already downloaded
    if output_path.exists():

        print(
            f"[{index}/{len(doc_names)}] "
            f"SKIP: {filename}"
        )

        skipped.append(
            filename
        )

        continue


    # URL encode filename safely
    encoded_filename = quote(
        filename
    )

    url = (
        f"{BASE_URL}/"
        f"{encoded_filename}"
    )


    print(
        f"[{index}/{len(doc_names)}] "
        f"Downloading: {filename}"
    )


    try:

        response = requests.get(
            url,
            timeout=120
        )


        if response.status_code == 200:

            output_path.write_bytes(
                response.content
            )

            size_mb = (
                len(response.content)
                / 1024
                / 1024
            )


            print(
                f"    OK "
                f"({size_mb:.2f} MB)"
            )

            success.append(
                filename
            )


        else:

            print(
                f"    FAILED "
                f"HTTP {response.status_code}"
            )

            failed.append(
                filename
            )


    except Exception as e:

        print(
            f"    ERROR: {e}"
        )

        failed.append(
            filename
        )


    # Be polite to GitHub
    time.sleep(0.3)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("Download Summary")
print("=" * 70)

print(
    "Downloaded:",
    len(success)
)

print(
    "Already existed:",
    len(skipped)
)

print(
    "Failed:",
    len(failed)
)


if failed:

    print(
        "\nFailed files:"
    )

    for filename in failed:

        print(
            filename
        )

else:

    print(
        "\nAll required FinanceBench PDFs are ready."
    )