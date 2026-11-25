# Evidence Retrieval from Financial Reports

A retrieval-augmented question-answering pipeline turns financial filings into searchable passages, retrieves candidate evidence with a dense index and reranks it. A separate evaluation compares retrieval with FinanceBench evidence pages.

## Explore the project

| File | What it contains |
| --- | --- |
| [PDF parsing](scripts/parse_pdf.py) | Parses filing pages and handles page-level extraction failures. |
| [Passage construction](scripts/build_chunks.py) | Detects sections and tables, reconstructs paragraphs and attaches page metadata. |
| [Retrieval and reranking](scripts/pipeline.py) | BGE embeddings, FAISS search and second-stage reranking. |
| [Evidence retrieval evaluation](scripts/evaluate_quick.py) | Scores retrieved passages against annotated evidence pages. |
| [Answer generation](scripts/chat.py) | Builds a context from selected passages and calls the configured language model. |

## Data and use

The included FinanceBench question file is the open-source subset used by the evaluation. Filing downloads, indexes and API credentials are external inputs. Build indexes from the project directory using the original scripts.
