import fitz
import pdfplumber

pdf_path = "/Users/vaibhavkrishali/Desktop/projects/RAG over pdf DRDO/code/data/pdfs/mil-std-810h.pdf"

with fitz.open(pdf_path) as doc:
    print(f"Page count: {doc.page_count}")

    # PyMuPDF uses zero-based page indexes, but PDF viewers show one-based page
    # numbers. Store page_index + 1 as page_number in all future metadata.
    """ 
   print("\nPages 100-110 summary:")
    start_page = 100
    end_page = min(110, doc.page_count - 1)
    for page_index in range(start_page, end_page + 1):
        page_number = page_index + 1
        page = doc[page_index]
        text_blocks = page.get_text("blocks")
        images = page.get_images()
        print(
            f"page {page_number} (index {page_index}): "
            f"{len(text_blocks)} text blocks, {len(images)} images"
        )

    print("\nPages with images:")
    first_image_page_index = None
    for page_index, page in enumerate(doc):
        images = page.get_images()
        if images:
            page_number = page_index + 1
            print(f"page {page_number} (index {page_index}): {len(images)} images")
            if first_image_page_index is None:
                first_image_page_index = page_index

    if first_image_page_index is not None:
        page_number = first_image_page_index + 1
        page = doc[first_image_page_index]
        blocks = page.get_text("blocks")
        print(f"\nFull block list for page {page_number} (index {first_image_page_index}):")
        print(blocks)
    else:
        print("\nNo pages with images found.")
    
    info = doc[601].get_image_info()
    print(len(info))
    for im in info:
        print(im["bbox"])
    
    start_page_number = 500
    end_page_number = 600
    print("Page numbers: [ ")
    for page_number in range(start_page_number, end_page_number + 1):
        page_index = page_number - 1
        page = doc[page_index]
        table_finder = page.find_tables()
        tables = table_finder.tables
        if tables:
            print(f"{page_number} (index {page_index}), tables: {len(tables)}")
    print(" ]")
    """
    # Diagnostic: page 33 (eyeballed) has 1 table with 3 rows x 2 columns
    # including the header. Our previous loop reported 3 rows x 6 columns.
    # Run BOTH detectors on this one page and print every row so we can see
    # what each one "sees" versus what is actually on the page.
    page_number = 40
    page_index = page_number - 1

    print(f"\n=== fitz.find_tables() on page {page_number} (index {page_index}) ===")
    fitz_tables = doc[page_index].find_tables().tables
    print(f"tables found: {len(fitz_tables)}")
    for t_idx, table in enumerate(fitz_tables):
        rows = table.extract()
        print(f"\n  table {t_idx}: {len(rows)} rows")
        print(f"  row lengths: {[len(r) for r in rows]}")
        for r_idx, row in enumerate(rows):
            print(f"    row {r_idx}: {row}")

print(f"\n=== pdfplumber.extract_tables() on page {page_number} (index {page_index}) ===")
with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[page_index]

    # What does pdfplumber actually SEE on this page? If lines is empty but
    # rects has objects, the visible "borders" are drawn as rectangles, not
    # line strokes -- and vertical_strategy="lines" will fall back to text.
    print(f"  page.lines : {len(page.lines)}")
    print(f"  page.rects : {len(page.rects)}")
    print(f"  page.edges : {len(page.edges)}")
    print(f"  page.curves: {len(page.curves)}")

    # Pass 1: default settings (whatever pdfplumber chooses on its own).
    print("\n  -- default settings --")
    for t_idx, rows in enumerate(page.extract_tables()):
        print(f"  table {t_idx}: {len(rows)} rows, row lengths {[len(r) for r in rows]}")
        for r_idx, row in enumerate(rows):
            print(f"    row {r_idx}: {row}")

    # Pass 2: force lines-only strategy. If the borders are real line objects,
    # this should give us 2 columns. If it gives the same 6-column result,
    # the borders aren't being detected as lines at all.
    print("\n  -- forced vertical_strategy='lines' --")
    settings = {"vertical_strategy": "lines", "horizontal_strategy": "lines"}
    for t_idx, rows in enumerate(page.extract_tables(table_settings=settings)):
        print(f"  table {t_idx}: {len(rows)} rows, row lengths {[len(r) for r in rows]}")
        for r_idx, row in enumerate(rows):
            print(f"    row {r_idx}: {row}")
