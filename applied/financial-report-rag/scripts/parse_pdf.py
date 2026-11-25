import json
import sys
from pathlib import Path

import pymupdf
import pymupdf4llm


# ============================================================
# Fallback: plain PyMuPDF text extraction
# ============================================================

def plain_text_fallback(doc, page_index, document_name):

    page = doc[page_index]

    text = page.get_text(
        "text",
        sort=True
    )

    return {
        "metadata": {
            "document": document_name,
            "page_number": page_index + 1,
            "page_count": len(doc),
            "parse_method": "pymupdf_plain_text"
        },
        "page_boxes": [],
        "text": text
    }


# ============================================================
# Recover one page
# ============================================================

def parse_single_page(
    pdf_path,
    doc,
    page_index,
    document_name
):

    page_number = page_index + 1

    # --------------------------------------------------------
    # Attempt 1:
    # PyMuPDF4LLM default table strategy
    # --------------------------------------------------------

    try:

        result = pymupdf4llm.to_markdown(
            pdf_path,
            pages=[page_index],
            page_chunks=True,
            use_ocr=False
        )

        if result:

            page = result[0]

            if "metadata" not in page:
                page["metadata"] = {}

            page["metadata"]["document"] = document_name
            page["metadata"]["page_number"] = page_number
            page["metadata"]["parse_method"] = (
                "pymupdf4llm"
            )

            return page

    except Exception as e:

        print(
            f"  Page {page_number}: "
            f"default parser failed: {e}"
        )


    # --------------------------------------------------------
    # Attempt 2:
    # Text-based table detection
    #
    # Useful when line/grid detection crashes.
    # --------------------------------------------------------

    try:

        print(
            f"  Page {page_number}: "
            f"retrying with table_strategy='text'..."
        )

        result = pymupdf4llm.to_markdown(
            pdf_path,
            pages=[page_index],
            page_chunks=True,
            use_ocr=False,
            table_strategy="text"
        )

        if result:

            page = result[0]

            if "metadata" not in page:
                page["metadata"] = {}

            page["metadata"]["document"] = document_name
            page["metadata"]["page_number"] = page_number
            page["metadata"]["parse_method"] = (
                "pymupdf4llm_table_text"
            )

            return page

    except Exception as e:

        print(
            f"  Page {page_number}: "
            f"text table parser failed: {e}"
        )


    # --------------------------------------------------------
    # Attempt 3:
    # Plain text extraction
    # --------------------------------------------------------

    print(
        f"  Page {page_number}: "
        f"falling back to plain PyMuPDF text"
    )

    return plain_text_fallback(
        doc,
        page_index,
        document_name
    )


# ============================================================
# Full-document recovery
# ============================================================

def recover_page_by_page(
    pdf_path,
    document_name
):

    print(
        "\nFull-document parsing failed."
    )

    print(
        "Starting page-by-page recovery..."
    )

    doc = pymupdf.open(
        pdf_path
    )

    pages = []

    fallback_pages = []


    try:

        total_pages = len(doc)

        for page_index in range(
            total_pages
        ):

            page_number = (
                page_index + 1
            )

            print(
                f"Recovering page "
                f"{page_number}/{total_pages}"
            )

            page = parse_single_page(
                pdf_path,
                doc,
                page_index,
                document_name
            )


            parse_method = (
                page
                .get("metadata", {})
                .get(
                    "parse_method",
                    "unknown"
                )
            )


            if parse_method != "pymupdf4llm":

                fallback_pages.append(
                    {
                        "page": page_number,
                        "method": parse_method
                    }
                )


            pages.append(
                page
            )

    finally:

        doc.close()


    return pages, fallback_pages


# ============================================================
# Main
# ============================================================

def main():

    if len(sys.argv) < 2:

        raise ValueError(
            "Usage: "
            "python parse_pdf.py "
            "<document_name>"
        )


    document_name = (
        sys.argv[1]
    )


    pdf_path = Path(
        f"data/raw/{document_name}.pdf"
    )

    output_path = Path(
        f"data/processed/"
        f"{document_name}_pages.json"
    )


    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )


    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        "Parsing:",
        pdf_path
    )


    fallback_pages = []


    # ========================================================
    # First choice:
    # parse entire document normally
    # ========================================================

    try:

        pages = pymupdf4llm.to_markdown(
            str(pdf_path),
            page_chunks=True,
            use_ocr=False
        )


        for page in pages:

            if "metadata" not in page:
                page["metadata"] = {}


            page["metadata"][
                "document"
            ] = document_name


            page["metadata"][
                "parse_method"
            ] = "pymupdf4llm"


        print(
            "Full-document parsing succeeded."
        )


    # ========================================================
    # If any PyMuPDF4LLM layout bug occurs,
    # recover page by page.
    # ========================================================

    except Exception as e:

        print(
            "\nWARNING:"
        )

        print(
            "PyMuPDF4LLM full-document "
            "parsing failed."
        )

        print(
            "Reason:",
            repr(e)
        )


        pages, fallback_pages = (
            recover_page_by_page(
                str(pdf_path),
                document_name
            )
        )


    # ========================================================
    # Save
    # ========================================================

    print(
        f"\nParsed {len(pages)} pages"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            pages,
            f,
            ensure_ascii=False,
            indent=2
        )


    print(
        "Saved:",
        output_path
    )


    # ========================================================
    # Recovery report
    # ========================================================

    if fallback_pages:

        print(
            "\nFallback pages:"
        )

        for item in fallback_pages:

            print(
                f"  Page {item['page']} "
                f"-> {item['method']}"
            )

    else:

        print(
            "\nFallback pages: 0"
        )


if __name__ == "__main__":
    main()