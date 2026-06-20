"""Throwaway probe: render the FAILING table regions to images (reusing pdfplumber
bboxes + fitz clip-render) and see what the local VLM transcribes. Compare to the
known-correct value. Cheap de-risk before building a VLM table pipeline."""
import fitz
import pdfplumber
import ollama
from ingest import PDF_PATH

VLM = "gemma3:4b"
# page -> the fact the text pipeline got WRONG (our target to recover)
TARGETS = {
    1075: "Air pressure/temp table; 50,000 ft -> 10 kPa, -76 C (text said -105)",
    543:  "Table 515.8A-IV; Air-to-Air Missile q max = 1600 (text said 1602)",
    400:  "Category-4 vibration table; 7 Hz -> ASD 0.30000 (text abstained)",
    48:   "Table G Shipboard locations incl. Masthead (text answered wrong table)",
}
PROMPT = ("Transcribe this table to GitHub-flavored markdown. Preserve every row, "
          "column and header exactly as shown. Output only the table.")

fdoc = fitz.open(PDF_PATH)
pdf = pdfplumber.open(PDF_PATH)

for pno, target in TARGETS.items():
    ppage = pdf.pages[pno - 1]
    fpage = fdoc[pno - 1]
    tables = list(ppage.find_tables())
    print(f"\n{'='*72}\nPAGE {pno} ({len(tables)} table region(s)) -- TARGET: {target}")
    for ti, t in enumerate(tables):
        x0, top, x1, bottom = t.bbox
        rect = fitz.Rect(x0 - 4, top - 4, x1 + 4, bottom + 4)          # small pad
        png = fpage.get_pixmap(clip=rect, dpi=200).tobytes("png")
        path = f"/tmp/tbl_p{pno}_t{ti}.png"
        open(path, "wb").write(png)
        resp = ollama.chat(model=VLM, options={"temperature": 0},
                           messages=[{"role": "user", "content": PROMPT,
                                      "images": [path]}])
        print(f"\n--- table {ti} bbox=({x0:.0f},{top:.0f},{x1:.0f},{bottom:.0f}) "
              f"img={path} ---")
        print(resp["message"]["content"][:900])

fdoc.close(); pdf.close()
