#!/usr/bin/env python3
import os
import pickle
import requests
from readability import Document
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from dotenv import load_dotenv
import pdfplumber
import tiktoken

# ─── Configuration ────────────────────────────────────────────────────────────
load_dotenv()
KB_FOLDER = os.path.join(os.path.dirname(__file__), "kb_chunks")
os.makedirs(KB_FOLDER, exist_ok=True)

# ─── Full list of pages to scrape ─────────────────────────────────────────────
URLS = [
    "https://www.ripleycourt.co.uk/",
    "https://www.ripleycourt.co.uk/19/welcome-from-the-headmaster",
    "https://www.ripleycourt.co.uk/20/an-introduction-to-ripley-court-school",
    "https://www.ripleycourt.co.uk/972/what-our-parents-say-about-us",
    "https://www.ripleycourt.co.uk/734/our-campus",
    "https://www.ripleycourt.co.uk/21/aims-ethos-and-values",
    "https://www.ripleycourt.co.uk/22/history-of-the-school",
    "https://www.ripleycourt.co.uk/17/reeds-school",
    "https://www.ripleycourt.co.uk/24/results-and-destination-schools",
    "https://www.ripleycourt.co.uk/25/inspection-reports",
    "https://www.ripleycourt.co.uk/130/staff-members",
    "https://www.ripleycourt.co.uk/74/ripley-court-parent-teacher-association-rcpta",
    "https://www.ripleycourt.co.uk/29/ripley-court-location",
    "https://www.ripleycourt.co.uk/73/policies/category/21/school-policies",
    "https://www.ripleycourt.co.uk/63/term-dates",
    "https://www.ripleycourt.co.uk/39/pastoral-care",
    "https://www.ripleycourt.co.uk/38/sport",
    "https://www.ripleycourt.co.uk/174/digital-learning",
    "https://www.ripleycourt.co.uk/213/computer-science-and-library",
    "https://www.ripleycourt.co.uk/41/learning-enrichment",
    "https://www.ripleycourt.co.uk/35/forest-school",
    "https://www.ripleycourt.co.uk/33/pre-prep",
    "https://www.ripleycourt.co.uk/28/outreach-programme",
    "https://www.ripleycourt.co.uk/741/weekly-enrichment",
    "https://www.ripleycourt.co.uk/421/clubs",
    "https://www.ripleycourt.co.uk/66/holiday-clubs",
    "https://www.ripleycourt.co.uk/42/school-hours-after-school-care",
    "https://www.ripleycourt.co.uk/75/uniform",
    "https://www.ripleycourt.co.uk/72/menus",
    "https://www.ripleycourt.co.uk/67/venue-hire",
    "https://www.ripleycourt.co.uk/46/prospectus",
    "https://www.ripleycourt.co.uk/45/open-mornings-and-visits",
    "https://www.ripleycourt.co.uk/47/enquire",
    "https://www.ripleycourt.co.uk/48/begin-the-registration-process",
    "https://www.ripleycourt.co.uk/49/admissions-faqs",
    "https://www.ripleycourt.co.uk/50/fees",
    "https://www.ripleycourt.co.uk/625/scholarships-and-bursaries",
    "https://www.ripleycourt.co.uk/54/little-twigs-stay-play",
    "https://www.ripleycourt.co.uk/32/nursery",
    "https://www.ripleycourt.co.uk/1425/head-of-early-years-and-ks1",
    "https://www.ripleycourt.co.uk/34/upper-court-prep",
    "https://www.ripleycourt.co.uk/901/prep-school-1",
    "https://www.ripleycourt.co.uk/36/curriculum",
    "https://www.ripleycourt.co.uk/221/academics/subject/16/art",
    "https://www.ripleycourt.co.uk/221/academics/subject/8/computing",
    "https://www.ripleycourt.co.uk/221/academics/subject/17/drama",
    "https://www.ripleycourt.co.uk/221/academics/subject/1/english",
    "https://www.ripleycourt.co.uk/221/academics/subject/15/food-technology",
    "https://www.ripleycourt.co.uk/221/academics/subject/10/french-modern-foreign-languages",
    "https://www.ripleycourt.co.uk/221/academics/subject/5/geography",
    "https://www.ripleycourt.co.uk/221/academics/subject/4/history",
    "https://www.ripleycourt.co.uk/221/academics/subject/12/learning-support",
    "https://www.ripleycourt.co.uk/221/academics/subject/2/mathematics",
    "https://www.ripleycourt.co.uk/221/academics/subject/11/music",
    "https://www.ripleycourt.co.uk/221/academics/subject/7/philosophy",
    "https://www.ripleycourt.co.uk/221/academics/subject/13/pshee",
    "https://www.ripleycourt.co.uk/221/academics/subject/6/religious-studies",
    "https://www.ripleycourt.co.uk/221/academics/subject/3/science",
    "https://www.ripleycourt.co.uk/221/academics/subject/14/sport-pe",
    "https://www.ripleycourt.co.uk/221/academics/subject/9/verbal-non-verbal-reasoning",
    "https://www.ripleycourt.co.uk/44/visit-us",
    "https://www.ripleycourt.co.uk/55/latest-news",
    "https://www.ripleycourt.co.uk/64/calendar",
    "https://www.ripleycourt.co.uk/56/newsletters",
    "https://schoolbase.online/Logon?DName=Reeds",
    "https://www.ripleycourt.co.uk/136/alumni",
    "https://ripleycourtshop.co.uk/",
    "https://ripleycourtshop.co.uk/pages/uniform-list",
    "https://ripleycourtshop.co.uk/pages/contact",
    "https://ripleycourtshop.co.uk/collections/boy-nursery-transition",
    "https://ripleycourtshop.co.uk/collections/boy-reception-year-2",
    "https://ripleycourtshop.co.uk/collections/boy-year-3-6",
    "https://ripleycourtshop.co.uk/collections/girl-nursery-transition",
    "https://ripleycourtshop.co.uk/collections/girl-reception-year-2",
    "https://ripleycourtshop.co.uk/collections/girl-year-3-6",
    "https://www.ripleycourt.co.uk/16/current-vacancies",
]


# ─── Chunking parameters ──────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
tokenizer     = tiktoken.get_encoding("cl100k_base")

def text_to_chunks(text):
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + CHUNK_SIZE
        chunks.append(tokenizer.decode(tokens[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# ─── Scrape & chunk ─────────────────────────────────────────────────────────—
metadata = []

for url in URLS:
    print("Fetching:", url)
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
    except Exception as e:
        print("  ⚠️ fetch failed:", e)
        continue

    # Extract full text
    if url.lower().endswith(".pdf"):
        try:
            with open("temp.pdf", "wb") as f:
                f.write(res.content)
            text = ""
            with pdfplumber.open("temp.pdf") as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        finally:
            os.remove("temp.pdf")
        full_text = text
    else:
        soup_full = BeautifulSoup(res.text, "html.parser")
        tables = []
        for table in soup_full.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cols = [td.get_text(strip=True) for td in tr.find_all(["td","th"])]
                if cols:
                    rows.append(" | ".join(cols))
            if rows:
                tables.append("TABLE:\n" + "\n".join(rows))

        doc = Document(res.text)
        article_html = doc.summary()
        article_text = BeautifulSoup(article_html, "html.parser").get_text(separator="\n").strip()
        full_text = "\n\n".join(tables + [article_text])

    # Chunking
    chunks = text_to_chunks(full_text)
    print(f"  → {len(chunks)} chunks")

    for chunk in chunks:
        metadata.append({"text": chunk, "url": url})

# ─── Save metadata.pkl ─────────────────────────────────────────────────────────
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
print(f"Saved metadata ({len(metadata)} chunks) to metadata.pkl")
