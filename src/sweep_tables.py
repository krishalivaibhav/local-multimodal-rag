"""
Sweep every page of MIL-STD-810H, run pdfplumber.extract_tables, and report
how many tables look MALFORMED.

Goal of this script: measurement only. Do not fix, drop, or merge anything --
we only want a count so we can decide what to do about it.

Run with:
    python src/sweep_tables.py
"""

import pdfplumber

PDF_PATH = "/Users/vaibhavkrishali/Desktop/projects/RAG over pdf DRDO/code/data/pdfs/mil-std-810h.pdf"


def _nonempty_cols(row):
    """Return the set of column indices in `row` that hold real content."""
    return {i for i, cell in enumerate(row) if cell and str(cell).strip()}


def malformed_reason(rows):
    """
    Return a reason string if the table looks malformed, else None.

    Reasons:
      "too_few_rows"  -- fewer than 2 rows; cannot validate
      "empty_content" -- header or data has no non-empty cells; expected on
                         flowchart-box false positives (finding #9)
      "col_disjoint"  -- header columns and data columns are disjoint;
                         the page-33 / page-40 phantom-column pattern
      "single_col"    -- fewer than 2 columns hold any data across the whole
                         table; a real table has >=2 content columns, so this
                         is a single-column box (flowchart node, list), not a
                         table (finding #9, page 9 leak)
    """
    if len(rows) < 2:
        return "too_few_rows"

    header_cols = _nonempty_cols(rows[0])
    data_cols = set()
    for row in rows[1:]:
        data_cols |= _nonempty_cols(row)

    if not header_cols or not data_cols:
        return "empty_content"

    if header_cols.isdisjoint(data_cols):
        return "col_disjoint"

    if len(header_cols | data_cols) < 2:
        return "single_col"

    return None


def main():
    total_tables = 0
    reason_counts = {"too_few_rows": 0, "empty_content": 0, "col_disjoint": 0}
    examples_by_reason = {"too_few_rows": [], "empty_content": [], "col_disjoint": []}
    EXAMPLES_PER_REASON = 3

    with pdfplumber.open(PDF_PATH) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1  # 1-based, per finding #4

            tables = page.extract_tables()
            if not tables:
                continue

            for table_idx, rows in enumerate(tables):
                total_tables += 1
                col_count = len(rows[0]) if rows else 0

                reason = malformed_reason(rows)
                if reason is None:
                    continue

                reason_counts[reason] += 1
                if len(examples_by_reason[reason]) < EXAMPLES_PER_REASON:
                    examples_by_reason[reason].append({
                        "page": page_number,
                        "table_idx": table_idx,
                        "row_count": len(rows),
                        "col_count": col_count,
                        "row_0": rows[0],
                        "row_1": rows[1] if len(rows) > 1 else None,
                    })

    total_malformed = sum(reason_counts.values())
    well_formed = total_tables - total_malformed

    print(f"\n=== SWEEP SUMMARY ===")
    print(f"Total tables found        : {total_tables}")
    print(f"Well-formed               : {well_formed}")
    print(f"Malformed (any reason)    : {total_malformed}")
    if total_tables:
        for reason, count in reason_counts.items():
            pct = 100 * count / total_tables
            print(f"  {reason:<15s}        : {count} ({pct:.1f}%)")

    for reason, examples in examples_by_reason.items():
        if not examples:
            continue
        print(f"\n=== EXAMPLES: {reason} ({len(examples)} shown) ===")
        for ex in examples:
            print(f"\npage {ex['page']}, table {ex['table_idx']}: "
                  f"{ex['row_count']} rows x {ex['col_count']} cols")
            print(f"  row 0: {ex['row_0']}")
            print(f"  row 1: {ex['row_1']}")


if __name__ == "__main__":
    main()
