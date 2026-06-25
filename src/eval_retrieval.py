"""
eval_retrieval.py -- measure retrieval quality before adding an LLM.

Baseline-first: an LLM can paper over weak retrieval, so we measure retrieval
on its own. Metrics (page-level correctness):
  Hit@k -- fraction of questions whose expected page appears in the top-k.
  MRR   -- mean of 1/rank of the first correct hit (0 if not in top-k).

The TEST_CASES list IS the ground truth. Each pair must be VERIFIED against the
actual PDF -- a wrong expected_page makes every metric meaningless. Seeded with
a few cases confirmed during the build; YOU expand it (see TODO).
"""

from query import retrieve, route_query

# Each case: question -> the 1-based page where its answer actually lives.
# 'type' is the chunk type the answer SHOULD come from (diagnostic, not enforced).
# 'expected_answer_contains' = verified key fact(s) the answer must contain (ALL of
#   them, case-insensitive substring) -- used by eval_answer.py for keyword-match
#   value grading. Keys are format-stable facts (numbers / identifiers / distinctive
#   words), never paraphrasable prose, to limit brittleness.
# 'note' records how the answer was verified so the test set stays auditable.
# All cases below were verified by reading the page in the actual PDF.
TEST_CASES = [
    # --- prose ---
    {"question": "Which NATO STANAG and AECTP method was this test method adapted from?",
     "expected_page": 846, "type": "prose",
     "expected_answer_contains": ["4370", "315"],
     "note": "adapted from NATO STANAG 4370, AECTP 300, Method 315"},
    {"question": "When computing the Cholesky factor L, what must be done if a zero element is found on the diagonal?",
     "expected_page": 995, "type": "prose",
     "expected_answer_contains": ["column"],
     "note": "positive semi-definite L computation; entire column set to zero"},
    {"question": "What does Procedure II of the loose cargo transportation method cover?",
     "expected_page": 361, "type": "prose",
     "expected_answer_contains": ["truck"],
     "note": "4.4.2 Procedure II - Loose cargo transportation (materiel carried in/on trucks)"},
    {"question": "What causes the shock in Torpedo Launch Immersion?",
     "expected_page": 70, "type": "prose",
     "expected_answer_contains": ["boost"],
     "note": "Torpedo Launch Immersion - shock due to launch boost"},
    {"question": "What is the fourth, specialized form of test tolerance mentioned with Gunfire Shock Method 519.8?",
     "expected_page": 867, "type": "prose",
     "expected_answer_contains": ["periodic"],
     "note": "periodic repetition of an event, tolerance from time-trace ensemble statistics"},

    # --- table_row (well-formed -> Format Y) ---
    {"question": "what does ASD(Peak) mean?",
     "expected_page": 501, "type": "table_row",
     "expected_answer_contains": ["maximum"],
     "note": "page 501 ASD nomenclature table (original smoke-query case); ASD(Peak) = max amplitude"},
    {"question": "What does ASD(Stdv) mean and how is it calculated?",
     "expected_page": 501, "type": "table_row",
     "expected_answer_contains": ["standard deviation"],
     "note": "page 501 ASD table, row 3: standard deviation of each spectral line over n averages"},
    {"question": "Under what condition is the dynamic pressure calculation valid with respect to Mach number?",
     "expected_page": 435, "type": "table_row",
     "expected_answer_contains": ["1.0"],
     "note": "notes row: valid only for Mach numbers less than 1.0"},
    {"question": "What is the Peak Acceleration (G-Pk) for the Functional Test for Ground Equipment?",
     "expected_page": 564, "type": "table_row",
     "expected_answer_contains": ["40"],
     "note": "page 564 table, Functional Test for Ground Equipment, G-Pk = 40"},

    # --- table_flat (col_disjoint -> Format B) ---
    {"question": "Which DID number covers Mechanical Vibrations of Shipboard Equipment?",
     "expected_page": 33, "type": "table_flat",
     "expected_answer_contains": ["DI-ENVR-81647"],
     "note": "DI-ENVR-81647, page 33 DID table (flattened)"},
    {"question": "What preheat temperature corresponds to an air pressure of 83 kPa (12 psi)?",
     "expected_page": 679, "type": "table_flat",
     "expected_answer_contains": ["46"],
     "note": "page 679 atomizing table: 83 kPa -> 46 C (115 F)"},
    {"question": "In the chemical agents table, what are the possible uses listed for Aircraft Cleaners?",
     "expected_page": 141, "type": "table_flat",
     "expected_answer_contains": ["ground equipment"],
     "note": "page 141 chemical/source/uses table, Aircraft Cleaners row (uses: Ground Equipment/...)"},
    {"question": "What percent of particles have an effective diameter of 71 to 82 micrometers?",
     "expected_page": 1066, "type": "table_flat",
     "expected_answer_contains": ["22"],
     "note": "page 1066 particle-size distribution: 71-82 um -> 22 percent"},
]

# Harder, more INDEPENDENT test set. Authored from reading raw PDF pages (not the
# chunk store), so answers are not retrievable by construction -- this set can target
# content we chunk badly and span the whole document. Hybrid provenance: cases 1-15
# drafted by Claude from the source + user-verified; cases 16-20 are user-authored
# "blind" (Claude never saw the source) for an unbiased spot-check. Several are
# deliberate stress-cases on known-weak areas (marked in 'note').
HARD_CASES = [
    # --- straightforward factual, new regions ---
    {"question": "In Procedure I-A (thermal shock), within how long must the test item be transferred to the second temperature?",
     "expected_page": 130, "type": "prose", "expected_answer_contains": ["minute"],
     "note": "Method 503.7 Procedure I-A Step 2: no more than one minute"},
    {"question": "How long must all tactical vehicles and guns be able to deep ford?",
     "expected_page": 300, "type": "prose", "expected_answer_contains": ["six"],
     "note": "Method 512.6: deep ford six (6) minutes"},
    {"question": "At an elevation of 50,000 feet, what is the temperature in degrees C (Table XXIV)?",
     "expected_page": 1075, "type": "table", "expected_answer_contains": ["-76"],
     "note": "Part Three Table XXIV: 15240 m / 50,000 ft -> -76 C"},
    {"question": "For mechanical shock, what is the default value for fmax?",
     "expected_page": 700, "type": "prose", "expected_answer_contains": ["10,000"],
     "note": "Method 519.8: default fmax = 10,000 Hz"},
    {"question": "What value of the ratio of specific heats used for air in cavity resonance testing?",
     "expected_page": 540, "type": "prose", "expected_answer_contains": ["1.4"],
     "note": "Method 515.8 Annex A: gamma = 1.4 for air"},
    {"question": "What approximate percentage of vehicle load capacity by weight should be used for realistic load configurations?",
     "expected_page": 388, "type": "prose", "expected_answer_contains": ["75"],
     "note": "Method 514.8 Annex C: ~75 percent of load capacity"},
    {"question": "Tailoring criteria should be taken not from natural environment data directly, but from what?",
     "expected_page": 60, "type": "prose", "expected_answer_contains": ["transformation"],
     "note": "Part One Annex C: from the transformations forcing functions create"},
    # --- paraphrased / figure-caption ---
    {"question": "How deep should fully enclosed armored vehicles be able to cross water?",
     "expected_page": 300, "type": "prose", "expected_answer_contains": ["turret"],
     "note": "Method 512.6 (paraphrased): deep ford to top of the turret"},
    {"question": "Which fighter aircraft is used as the example mission profile for altitude and Mach number?",
     "expected_page": 780, "type": "prose", "expected_answer_contains": ["F-15"],
     "note": "Method 520.5 Annex A figure caption: F-15 ground attack example"},
    # --- multi-fact (both parts must appear) ---
    {"question": "At 50,000 feet, what are both the air pressure (kPa) and the temperature (degrees C)?",
     "expected_page": 1075, "type": "table", "expected_answer_contains": ["10", "-76"],
     "note": "Table XXIV: 50,000 ft -> 10 kPa, -76 C"},
    # --- stress-cases on known-weak areas ---
    {"question": "What is the ASD (g2/Hz) at 7 Hz for the Category 4 composite wheeled-vehicle vibration exposure?",
     "expected_page": 400, "type": "table_flat", "expected_answer_contains": ["0.30000"],
     "note": "STRESS #14: Table 514.8C-VIII (table_flat) 7 Hz -> 0.30000"},
    {"question": "In Option 2 of the powered-on procedure, what does step b cover?",
     "expected_page": 774, "type": "pseudo_table", "expected_answer_contains": ["stabiliz"],
     "note": "STRESS #18: p774 Option 1|Option 2 pseudo-table, step b = Cold/Dry Stabilization"},
    {"question": "In Table G (Shipboard), name one location you specify durations for.",
     "expected_page": 48, "type": "table_flat", "expected_answer_contains": ["masthead"],
     "note": "STRESS #14: Part One Annex A Table G (form/table_flat): Masthead, exposed upper deck, ..."},
    {"question": "What does a q-q plot examine in the distribution of s(t)?",
     "expected_page": 906, "type": "prose",
     "expected_answer_contains": [("tails", "Gaussian", "normal")],
     "note": "STRESS math: Method 525.2 Annex B: examine the 'tails'/extreme values OR "
             "the fit to a Gaussian distribution -- either is a correct answer (adjudicated, #20)"},
    {"question": "Name a surface effect of warm, humid environments on materiel.",
     "expected_page": 210, "type": "prose", "expected_answer_contains": ["corrosion"],
     "note": "Method 507.6: oxidation and/or galvanic corrosion of metals"},
    # --- user-authored blind (Claude did not see the source) ---
    {"question": "In shock measurement system characterization, what minimum amplitude attenuation (dB) must the anti-alias filter provide?",
     "expected_page": 600, "type": "prose", "expected_answer_contains": ["50"],
     "note": "BLIND: min anti-alias filter attenuation = 50 dB"},
    {"question": "In the Beta Distribution speed example (Annex F), what is the Ratio of Avg/Max Speed?",
     "expected_page": 460, "type": "table_flat", "expected_answer_contains": ["64"],
     "note": "BLIND: beta distribution example, Ratio Avg/Max Speed = 64%"},
    {"question": "What temperature range (degrees C) is given for 'Aggravated solar' conditions in high-temperature testing?",
     "expected_page": 100, "type": "prose", "expected_answer_contains": ["71", "85"],
     "note": "BLIND: Aggravated solar -> 71 to 85 C"},
    {"question": "What is the recommended minimum rainfall rate for Procedure I (Rain and Blowing Rain)?",
     "expected_page": 196, "type": "prose", "expected_answer_contains": ["1.7"],
     "note": "BLIND: min rainfall rate = 1.7 mm/min"},
    {"question": "In Table 515.8A-IV, what is the q max value listed for an Air-to-Air Missile?",
     "expected_page": 543, "type": "table", "expected_answer_contains": ["1600"],
     "note": "BLIND: Table 515.8A-IV, Air-to-Air Missile q max = 1600"},
]

# Domain-adjacent unanswerable questions for the hard set (should abstain).
HARD_ABSTAIN_CASES = [
    {"question": "What is the unit cost of conducting a vibration test per this standard?",
     "note": "no costs in an environmental test standard"},
    {"question": "What encryption algorithm does this standard mandate for test data?",
     "note": "out of scope; no cryptography requirements"},
    {"question": "What is the maximum allowable cybersecurity vulnerability score for fielded materiel?",
     "note": "out of scope; not an environmental-test topic"},
]

# Multi-document set (finding #29): questions over the NASA handbook + MIL-STD-810G,
# each with expected_source_file so the eval checks the answer cites the RIGHT document
# (page numbers alone are ambiguous across docs -- 810G is a near-duplicate of 810H).
MULTIDOC_CASES = [
    # --- NASA Systems Engineering Handbook (cross-style content) ---
    {"question": "In NASA's project-type classification (Table 3.11-1), what kind of mission is Type A?",
     "expected_page": 48, "expected_source_file": "nasa_systems_engineering_handbook_0.pdf",
     "type": "table", "expected_answer_contains": ["Human Space Flight"],
     "note": "Table 3.11-1: Type A = Human Space Flight or Very Large Science/Robotic"},
    {"question": "What does the NASA SE competency Product Validation (SE 2.4) confirm the end product satisfies?",
     "expected_page": 25, "expected_source_file": "nasa_systems_engineering_handbook_0.pdf",
     "type": "prose", "expected_answer_contains": ["stakeholder expectations"],
     "note": "SE 2.4 product validation -> satisfies stakeholder expectations"},
    {"question": "The NASA glossary defines Reliability as the measure of the degree to which a system ensures what?",
     "expected_page": 200, "expected_source_file": "nasa_systems_engineering_handbook_0.pdf",
     "type": "prose", "expected_answer_contains": ["mission success"],
     "note": "glossary: reliability -> ensures mission success"},
    {"question": "What does CPI stand for in NASA's asset-protection planning?",
     "expected_page": 165, "expected_source_file": "nasa_systems_engineering_handbook_0.pdf",
     "type": "prose", "expected_answer_contains": ["Critical Program Information"],
     "note": "CPI = Critical Program Information"},
    {"question": "TPMs are used to confirm progress and identify what?",
     "expected_page": 72, "expected_source_file": "nasa_systems_engineering_handbook_0.pdf",
     "type": "prose", "expected_answer_contains": ["deficiencies"],
     "note": "TPMs confirm progress and identify deficiencies"},
    {"question": "What kind of test stands are listed among facilities to consider in Table 6.1-2?",
     "expected_page": 130, "expected_source_file": "nasa_systems_engineering_handbook_0.pdf",
     "type": "table", "expected_answer_contains": ["Propulsion"],
     "note": "Table 6.1-2 facilities include Propulsion Test Stands"},
    # --- MIL-STD-810G (near-duplicate of 810H; MUST cite 810G) ---
    {"question": "In MIL-STD-810G Method 512.5 (Immersion), what can the test verify in lieu of a rain test?",
     "expected_page": 230, "expected_source_file": "MIL-STD-810G.pdf",
     "type": "prose", "expected_answer_contains": ["watertightness"],
     "note": "DISAMBIG: 512.5 is G's numbering (810H = 512.6); immersion -> watertightness"},
    {"question": "In MIL-STD-810G, which method's Annex D gives operational tailoring guidance for vibration exposure?",
     "expected_page": 320, "expected_source_file": "MIL-STD-810G.pdf",
     "type": "prose", "expected_answer_contains": ["514.6"],
     "note": "DISAMBIG: 514.6 is G (810H = 514.8)"},
    {"question": "In MIL-STD-810G Method 505.5, what type of filter is presently preferred?",
     "expected_page": 140, "expected_source_file": "MIL-STD-810G.pdf",
     "type": "prose", "expected_answer_contains": ["glass"],
     "note": "Method 505.5 prefers glass filters"},
    {"question": "In MIL-STD-810G Table 516.6A-I, what is the normal tolerance factor for N=3 at gamma=0.50, beta=0.90?",
     "expected_page": 410, "expected_source_file": "MIL-STD-810G.pdf",
     "type": "table", "expected_answer_contains": ["1.50"],
     "note": "Table 516.6A-I: N=3, gamma=0.50, beta=0.90 -> 1.50"},
    {"question": "In MIL-STD-810G Part One Annex D, what is x(t) called relative to X(f)?",
     "expected_page": 60, "expected_source_file": "MIL-STD-810G.pdf",
     "type": "prose", "expected_answer_contains": ["inverse Fourier"],
     "note": "Annex D: x(t) = inverse Fourier transform of X(f)"},
]

# Figure-QA (image retrieval): questions authored from the ACTUAL rendered figures, NOT from the
# VLM descriptions (avoids the #16/#18 selection-bias trap). Measured against the VLM store
# (chroma_vlm), which carries figure_caption chunks. Some cases are EXPECTED to fail -- they
# honestly expose where gemma4 only PARTIALLY transcribes a complex diagram (finding #39).
FIGURE_CASES = [
    {"question": "In NASA's Lunar Sortie DRM example (Figure 4.1-3), what happens to the Earth Departure Stage and the Ascent Stage during the mission?",
     "expected_page": 61, "expected_source_file": "nasa_systems_engineering_handbook_0.pdf",
     "type": "figure", "expected_answer_contains": ["expended"],
     "note": "Fig 4.1-3 schematic: both stages are 'Expended' (VLM captured -> expect pass)"},
    {"question": "What method of atmospheric entry does the Lunar Sortie DRM (Figure 4.1-3) show for returning to Earth?",
     "expected_page": 61, "expected_source_file": "nasa_systems_engineering_handbook_0.pdf",
     "type": "figure", "expected_answer_contains": [("skip", "direct or skip")],
     "note": "Fig 4.1-3: 'Direct or Skip Land Entry' (VLM captured -> expect pass)"},
    {"question": "According to Figure 2.0-1 (SE in the Context of Overall Project Management), what are the System Design Processes?",
     "expected_page": 15, "expected_source_file": "nasa_systems_engineering_handbook_0.pdf",
     "type": "figure", "expected_answer_contains": ["Logical Decomposition", "Design Solution"],
     "note": "Fig 2.0-1: the 4 System Design Processes; gemma4 MISSED these (captured only 'Common') -> expect FAIL (honest negative)"},
    {"question": "What shared items are shown in the 'Common' center of Figure 2.0-1?",
     "expected_page": 15, "expected_source_file": "nasa_systems_engineering_handbook_0.pdf",
     "type": "figure", "expected_answer_contains": [("Stakeholders", "Risks")],
     "note": "Fig 2.0-1 'Common' center: Stakeholders, Risks, Config Mgmt, Data (VLM captured -> expect pass)"},
]

K = 5


def evaluate(test_cases, k=K):
    """Run each test case through retrieval and score it page-level.

    Primary metric is page-level (Hit@k / MRR): did a chunk from the right page
    surface, and how high? We also capture the source_type of the chunk that
    scored the hit, so we can see whether the EXPECTED kind of element (prose vs
    table_row vs table_flat) is the one actually winning -- a page hit off the
    wrong chunk type is a quieter kind of bug than a clean miss.
    """
    reciprocal_ranks = []
    hits = 0
    rows = []

    for case in test_cases:
        res = retrieve(case["question"], k=k, source_file=route_query(case["question"]))
        metas = res["metadatas"][0]
        pages = [m.get("page_number") for m in metas]
        types = [m.get("source_type") for m in metas]
        sfiles = [m.get("source_file") for m in metas]
        # match on (source_file, page): in a multi-doc store a page number alone is
        # ambiguous. Defaults to 810h so the original single-doc cases still work.
        exp_sf = case.get("expected_source_file", "mil-std-810h.pdf")

        rank = None
        for i, (page, sf) in enumerate(zip(pages, sfiles), start=1):
            if page == case["expected_page"] and sf == exp_sf:
                rank = i
                break

        # source_type of the chunk that produced the hit (None on a miss)
        hit_type = types[rank - 1] if rank is not None else None

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        rows.append((case, pages, types, rank, hit_type))

    n = len(test_cases)
    hit_at_k = hits / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0
    return hit_at_k, mrr, rows


def per_type_hit_at_k(rows):
    """Hit@k split by the expected chunk type -- shows which retrieval path is
    weak (e.g. table_flat questions missing more than prose)."""
    totals, hit = {}, {}
    for case, _pages, _types, rank, _hit_type in rows:
        t = case["type"]
        totals[t] = totals.get(t, 0) + 1
        if rank is not None:
            hit[t] = hit.get(t, 0) + 1
    return {t: (hit.get(t, 0), totals[t]) for t in totals}


if __name__ == "__main__":
    import sys

    # `python eval_retrieval.py hard|multidoc` selects the case set.
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    cases = {"hard": HARD_CASES, "multidoc": MULTIDOC_CASES,
             "figure": FIGURE_CASES}.get(arg, TEST_CASES)
    label = {"hard": "HARD", "multidoc": "MULTIDOC", "figure": "FIGURE"}.get(arg, "original")

    hit_at_k, mrr, rows = evaluate(cases)

    print(f"=== retrieval eval ({label} set, page-level, k={K}) ===")
    print(f"cases : {len(cases)}")
    print(f"Hit@{K} : {hit_at_k:.2f}")
    print(f"MRR   : {mrr:.3f}\n")

    print("Hit@k by expected type:")
    for t, (h, tot) in sorted(per_type_hit_at_k(rows).items()):
        print(f"  {t:<11s}: {h}/{tot}")
    print()

    for case, pages, types, rank, hit_type in rows:
        status = f"rank {rank}" if rank else "MISS"
        # flag when we hit the right page but off an unexpected chunk type
        type_flag = ""
        if rank is not None and hit_type != case["type"]:
            type_flag = f"  (!) hit via {hit_type}, expected {case['type']}"
        print(f"[{status:>7}] exp p{case['expected_page']} ({case['type']})"
              f" | got pages {pages}{type_flag}")
        print(f"           Q: {case['question']}")
