"""Throwaway: does table content also exist as a prose chunk on the same page?
For the 3 (!) cases, show top-8 hits (type+page) AND the table_flat chunk text."""
import chromadb
from query import search
from ingest import CHROMA_DIR, COLLECTION_NAME

client = chromadb.PersistentClient(path=CHROMA_DIR)
col = client.get_collection(COLLECTION_NAME)

CASES = [
    ("Which DID number covers Mechanical Vibrations of Shipboard Equipment?", 33),
    ("What preheat temperature corresponds to an air pressure of 83 kPa (12 psi)?", 679),
    ("What percent of particles have an effective diameter of 71 to 82 micrometers?", 1066),
]

for q, page in CASES:
    print(f"\n{'='*70}\nQ: {q}\nexpected page {page}")
    res = search(q, k=8)
    for rank, (m, d, dist) in enumerate(
            zip(res["metadatas"][0], res["documents"][0], res["distances"][0]), 1):
        mark = "  <== expected pg" if m.get("page_number") == page else ""
        print(f"  #{rank} {dist:.3f} [{m.get('source_type'):<10}] p{m.get('page_number')}{mark}")

    # what does the table_flat chunk for this page look like, and is it retrievable?
    flat = col.get(where={"$and": [{"page_number": page},
                                   {"source_type": "table_flat"}]},
                   include=["documents"])
    print(f"  -- table_flat chunk on p{page}: "
          f"{'NONE' if not flat['ids'] else repr(flat['documents'][0][:160])}")
    # the prose chunk(s) on this page that mention the answer
    prose = col.get(where={"$and": [{"page_number": page},
                                    {"source_type": "prose"}]},
                    include=["documents"])
    print(f"  -- prose chunks on p{page}: {len(prose['ids'])}")
