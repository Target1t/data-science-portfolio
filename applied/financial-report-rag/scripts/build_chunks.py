import json
import sys
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



# =====================================================
# Model
# =====================================================

semantic_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)



# =====================================================
# Financial table keywords
# =====================================================

TABLE_KEYWORDS = [

    "CONSOLIDATED STATEMENTS OF OPERATIONS",
    "CONSOLIDATED STATEMENT OF OPERATIONS",

    "CONSOLIDATED STATEMENTS OF CASH FLOWS",
    "CONSOLIDATED STATEMENT OF CASH FLOWS",

    "CONSOLIDATED BALANCE SHEETS",

    "NET INCOME",

    "TOTAL ASSETS",

    "TOTAL LIABILITIES",

    "NET SALES",

    "OPERATING ACTIVITIES",

    "INVESTING ACTIVITIES",

    "FINANCING ACTIVITIES",

]



# =====================================================
# Metadata
# =====================================================

def extract_company_year(doc):

    m=re.match(
        r"(.*)_(\d{4})_",
        doc
    )

    if m:

        return (
            m.group(1),
            int(m.group(2))
        )


    return "Unknown",None



# =====================================================
# Text cleaning
# =====================================================

def clean_text(text):

    lines=[]


    for line in text.split("\n"):


        line=line.strip()


        if not line:

            continue


        if line.lower()=="table of contents":

            continue


        if re.match(
            r"^\d+$",
            line
        ):

            continue


        lines.append(line)



    return "\n".join(lines)




# =====================================================
# Section detection
# =====================================================

def detect_section(text):

    t=text.upper()

    t=t.replace(
        " ",
        ""
    )


    if "CASHFLOWS" in t:

        return "cash_flow"


    if "STATEMENTSOFOPERATIONS" in t:

        return "income_statement"


    if "BALANCESHEETS" in t:

        return "balance_sheet"


    if "RISKFACTORS" in t:

        return "risk_factor"


    if "MANAGEMENTSDISCUSSION" in t:

        return "MD&A"


    return "unknown"





# =====================================================
# Table detector
# =====================================================

def is_financial_table(page):


    boxes=page.get(
        "page_boxes",
        []
    )


    has_table=False


    for b in boxes:

        if b.get("class")=="table":

            has_table=True



    if not has_table:

        return False



    text=page["text"].upper()



    return any(

        k in text

        for k in TABLE_KEYWORDS

    )





# =====================================================
# Paragraph reconstruction
# =====================================================

def reconstruct_paragraphs(text):


    text=clean_text(text)


    lines=[
        x.strip()
        for x in text.split("\n")
        if len(x.strip())>20
    ]


    paragraphs=[]

    buffer=[]



    for line in lines:


        buffer.append(line)



        joined=" ".join(buffer)



        # paragraph boundary

        if (

            line.endswith(".")

            or

            len(joined)>600

            or

            line.endswith(":")

        ):


            paragraphs.append(
                joined
            )


            buffer=[]



    if buffer:

        paragraphs.append(
            " ".join(buffer)
        )



    return paragraphs





# =====================================================
# Semantic chunking
# =====================================================

def semantic_chunk(

        text,

        similarity_threshold=0.55,

        max_chars=4000

):


    paragraphs=reconstruct_paragraphs(
        text
    )


    if len(paragraphs)==0:

        return []



    if len(paragraphs)==1:

        return paragraphs



    vectors=semantic_model.encode(

        paragraphs,

        normalize_embeddings=True

    )



    chunks=[]

    current=[]

    size=0



    for i,p in enumerate(paragraphs):


        current.append(p)

        size+=len(p)



        split=False



        if size>=max_chars:

            split=True



        if i < len(paragraphs)-1:


            sim=cosine_similarity(

                [vectors[i]],

                [vectors[i+1]]

            )[0][0]



            if sim < similarity_threshold:

                split=True



        if split:


            chunks.append(

                "\n".join(current)

            )


            current=[]

            size=0



    if current:

        chunks.append(
            "\n".join(current)
        )


    return chunks





# =====================================================
# Chunk builders
# =====================================================

def make_metadata(

        document,

        page,

        ctype,

        section

):


    company,year=extract_company_year(
        document
    )


    return {


        "company":company,

        "year":year,

        "document":document,

        "page":page,

        "type":ctype,

        "section":section

    }




def table_chunk(page,document):


    text=clean_text(
        page["text"]
    )


    return {

        "text":text,

        "metadata":

        make_metadata(

            document,

            page["metadata"]["page_number"],

            "financial_table",

            detect_section(text)

        )

    }





def narrative_chunks(page,document):


    results=[]


    chunks=semantic_chunk(
        page["text"]
    )



    for c in chunks:


        results.append({

            "text":c,


            "metadata":

            make_metadata(

                document,

                page["metadata"]["page_number"],

                "narrative",

                detect_section(c)

            )

        })


    return results





# =====================================================
# Main
# =====================================================

def main():


    document=sys.argv[1]


    input_file=(

        f"data/processed/"
        f"{document}_pages.json"

    )


    output_file=(

        f"data/processed/"
        f"{document}_chunks.json"

    )



    with open(
        input_file,
        encoding="utf-8"
    ) as f:

        pages=json.load(f)



    chunks=[]


    tables=0

    narratives=0



    for page in pages:


        if is_financial_table(page):


            chunks.append(

                table_chunk(
                    page,
                    document
                )

            )


            tables+=1


        else:


            cs=narrative_chunks(

                page,

                document

            )


            chunks.extend(cs)

            narratives+=len(cs)




    with open(

        output_file,

        "w",

        encoding="utf-8"

    ) as f:


        json.dump(

            chunks,

            f,

            ensure_ascii=False,

            indent=2

        )



    print("="*60)

    print(
        "Document:",
        document
    )

    print(
        "Total chunks:",
        len(chunks)
    )

    print(
        "Table chunks:",
        tables
    )

    print(
        "Narrative chunks:",
        narratives
    )

    print(
        "Saved:",
        output_file
    )

    print("="*60)



if __name__=="__main__":

    main()