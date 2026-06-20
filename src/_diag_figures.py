"""Throwaway: understand how figures appear as image blocks, to design region
clustering (finding #5: image xrefs != logical figures; p602 = 28 fragments)."""
import fitz
from config import PDF_PATH

doc = fitz.open(PDF_PATH)

# 1) corpus scan: which pages are image-heavy (candidate figure pages)?
counts = []
for i in range(doc.page_count):
    imgs = [b for b in doc[i].get_text("dict")["blocks"] if b.get("type") == 1]
    if imgs:
        counts.append((i + 1, len(imgs)))
counts.sort(key=lambda x: -x[1])
print("top image-heavy pages (1-based, #image-blocks):")
for pno, n in counts[:12]:
    print(f"  p{pno}: {n}")
print(f"\n{len(counts)} pages have >=1 image block; "
      f"{sum(1 for _, n in counts if n >= 5)} have >=5 (likely fragmented figures)")

# 2) detail on p602: bboxes of every image fragment + their bounding extent
print(f"\n{'='*60}\nPAGE 602 image fragments:")
page = doc[601]
img_blocks = [b for b in page.get_text("dict")["blocks"] if b.get("type") == 1]
xs0, ys0, xs1, ys1 = [], [], [], []
for b in img_blocks:
    x0, y0, x1, y1 = b["bbox"]
    xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
    print(f"  bbox=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})  "
          f"size={x1-x0:.0f}x{y1-y0:.0f}")
if img_blocks:
    print(f"\n  {len(img_blocks)} fragments span a combined region: "
          f"({min(xs0):.0f},{min(ys0):.0f},{max(xs1):.0f},{max(ys1):.0f})")
    print(f"  page size: {tuple(round(v) for v in page.rect)}")
