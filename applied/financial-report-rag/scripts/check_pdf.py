import fitz


pdf_path = "data/raw/apple_2024_10k.pdf"


doc = fitz.open(pdf_path)

print("页数:", doc.page_count)

page = doc[0]

text = page.get_text()

print(text[:500])