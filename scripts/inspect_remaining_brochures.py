import sys, json

sys.stdout.reconfigure(encoding='utf-8')

with open("data/all_29_extracted_raw.json", "r", encoding="utf-8") as f:
    docs = {d["filename"]: d for d in json.load(f)}

rem_files = [
    "Brochure - AllianceOne 4.0.pdf",
    "Cresindo-2k26-brochure (1).pdf",
    "Aicon'26 Brochure Digital File-Compressed (1).pdf",
    "Dhwani CET Brochure.pdf",
    "DSBA-BATTALIA-2K26-brochure.pdf",
    "FEST 2026 GUIDELINES.pdf",
    "IMPERIUM 2.0 - Brochure_.pdf",
    "Innovate-Ignite--26---Inter-Collegiate-brochure (2).pdf",
    "NATIONAL-LEGAL-HACKATHON-brochure.pdf",
    "PES PU Virinchi 2026 broucher 2008.pdf",
    "SARANSH 2026 Inter V.1.pdf"
]

for rf in rem_files:
    if rf in docs:
        print("=" * 80)
        print(f"DOCUMENT: {rf} (Pages: {docs[rf]['total_pages']})")
        print("=" * 80)
        # print first 3 pages
        for p in docs[rf]["pages"][:4]:
            print(f"--- Page {p['page_num']} ---")
            print(p["text"][:1000])
        print("\n")
