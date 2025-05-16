#!/usr/bin/env python3
import os
import pickle
import traceback
from datetime import date

import numpy as np
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from fuzzywuzzy import fuzz

# ─── Initialise ──────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise RuntimeError("OPENAI_API_KEY not set in .env")

# ─── URL lookup ───────────────────────────────────────────────────────────────
PAGE_LINKS = {
    "home":                               "https://www.ripleycourt.co.uk/",
    "welcome from the headmaster":        "https://www.ripleycourt.co.uk/19/welcome-from-the-headmaster",
    "introduction to ripley court":       "https://www.ripleycourt.co.uk/20/an-introduction-to-ripley-court-school",
    "what our parents say about us":      "https://www.ripleycourt.co.uk/972/what-our-parents-say-about-us",
    "our campus":                         "https://www.ripleycourt.co.uk/734/our-campus",
    "aims ethos and values":             "https://www.ripleycourt.co.uk/21/aims-ethos-and-values",
    "history of the school":              "https://www.ripleycourt.co.uk/22/history-of-the-school",
    "reeds school":                       "https://www.ripleycourt.co.uk/17/reeds-school",
    "results and destinations":           "https://www.ripleycourt.co.uk/24/results-and-destination-schools",
    "inspection reports":                 "https://www.ripleycourt.co.uk/25/inspection-reports",
    "staff members":                      "https://www.ripleycourt.co.uk/130/staff-members",
    "parent teacher association":         "https://www.ripleycourt.co.uk/74/ripley-court-parent-teacher-association-rcpta",
    "alumni":                             "https://www.ripleycourt.co.uk/136/alumni",
    "location":                           "https://www.ripleycourt.co.uk/29/ripley-court-location",
    "policies":                           "https://www.ripleycourt.co.uk/73/policies/category/21/school-policies",
    "term dates":                         "https://www.ripleycourt.co.uk/63/term-dates",
    "calendar":                           "https://www.ripleycourt.co.uk/64/calendar",
    "lunch":                              "https://www.ripleycourt.co.uk/72/menus",
    "sport":                              "https://www.ripleycourt.co.uk/38/sport",
    "uniform":                            "https://www.ripleycourt.co.uk/75/uniform",
    "prospectus":                         "https://www.ripleycourt.co.uk/46/prospectus",
    "enquire":                            "https://www.ripleycourt.co.uk/47/enquire",
    "registration deadlines":             "https://www.ripleycourt.co.uk/48/begin-the-registration-process",
    "fees":                               "https://www.ripleycourt.co.uk/50/fees",
    "scholarships and bursaries":         "https://www.ripleycourt.co.uk/625/scholarships-and-bursaries",
    # …add any others you need…
}

URL_LABELS = {
    PAGE_LINKS["prospectus"]:             "Download Prospectus",
    PAGE_LINKS["enquire"]:                "Enquire now",
    PAGE_LINKS["registration deadlines"]: "Begin Registration",
    PAGE_LINKS["fees"]:                   "View Fees",
    PAGE_LINKS["lunch"]:                  "View Menus",
    PAGE_LINKS["uniform"]:                "View Uniform",
    # …etc…
}

STATIC_QAS = {
    "prospectus": (
        "Please download our prospectus below to learn more about Ripley Court School.",
        PAGE_LINKS["prospectus"],
        URL_LABELS[PAGE_LINKS["prospectus"]]
    ),
    "enquire": (
        "You can enquire and we will tailor a prospectus exactly for you and your family.",
        PAGE_LINKS["enquire"],
        URL_LABELS[PAGE_LINKS["enquire"]]
    ),
    "fees": (
        "Our current fees and payment options are detailed on the Fees page.",
        PAGE_LINKS["fees"],
        URL_LABELS[PAGE_LINKS["fees"]]
    ),
    # …any other exact‐match Q&A you want…
}

# ─── Load embeddings & metadata ───────────────────────────────────────────────
with open("embeddings.pkl", "rb") as f:
    embeddings = np.stack(pickle.load(f), axis=0)
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

EMB_MODEL  = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"

# ─── System prompt ────────────────────────────────────────────────────────────
today = date.today().isoformat()
system_prompt = (
    f"You are a friendly, professional assistant for Ripley Court School.\n"
    f"Today's date is {today}.\n"
    "Begin with 'Thank you for your question!' and end with 'Anything else I can help you with today?'.\n"
    "If you do not know the answer, say 'I'm sorry, I don't have that information.'\n"
    "Use British spelling."
)

# ─── Create Flask app & enable CORS (serve static/chat.html) ─────────────────
app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/ask": {"origins": "*"}})

from flask import redirect

@app.route("/", methods=["GET"])
def home():
    # Redirect root URL to the static chat page
    return redirect("/static/chat.html")


# ─── Helper functions ────────────────────────────────────────────────────────
def cosine_similarities(matrix, vector):
    dot = matrix @ vector
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    return dot / (norms + 1e-8)

def remove_bullets(text):
    return " ".join(
        line[2:].strip() if line.startswith("- ") else line.strip()
        for line in text.split("\n")
    )

def format_response(ans):
    footer = "Anything else I can help you with today?"
    ans = ans.replace(footer, "").strip()
    sents, paras, curr = ans.split(". "), [], []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        curr.append(s.rstrip("."))
        if len(curr) >= 3 or s.endswith("?"):
            paras.append(". ".join(curr) + ".")
            curr = []
    if curr:
        paras.append(". ".join(curr) + ".")
    if not paras or not paras[0].startswith("Thank you for your question"):
        paras.insert(0, "Thank you for your question!")
    paras.append(footer)
    return "\n\n".join(paras)

# ─── /ask endpoint ────────────────────────────────────────────────────────────
@app.route("/ask", methods=["POST"])
@cross_origin()
def ask():
    try:
        data     = request.get_json(force=True)
        question = data.get("question","").strip()
        if not question:
            return jsonify(error="No question provided"), 400

        key = question.lower().rstrip("?")

        # 1) Exact static
        if key in STATIC_QAS:
            raw,url,label = STATIC_QAS[key]
            return jsonify(
                answer=format_response(remove_bullets(raw)),
                url=url,
                link_label=label
            ), 200

        # 2) Fuzzy static
        for sk,(raw,url,label) in STATIC_QAS.items():
            if fuzz.partial_ratio(sk, key) > 80:
                return jsonify(
                    answer=format_response(remove_bullets(raw)),
                    url=url,
                    link_label=label
                ), 200

        # 3) Welcome trigger
        if question == "__welcome__":
            raw = (
                "Hi there! Ask me anything about Ripley Court School.\n\n"
                "We tailor our prospectus to your enquiry. For more details, visit below.\n\n"
                "Anything else I can help you with today?"
            )
            return jsonify(
                answer=remove_bullets(raw),
                url=PAGE_LINKS["enquire"],
                link_label=URL_LABELS[PAGE_LINKS["enquire"]]
            ), 200

        # 4) Guard “how many…”
        if key.startswith("how many"):
            return jsonify(
                answer=format_response("I'm sorry, I don't have that information."),
                url=None
            ), 200

        # 5) Keyword → URL
        relevant_url = None
        for k,u in PAGE_LINKS.items():
            if k in key or any(fuzz.partial_ratio(k,w)>80 for w in key.split() if len(w)>3):
                relevant_url = u
                break

        # 6) RAG fallback
        emb = openai.embeddings.create(model=EMB_MODEL, input=question)
        q_vec = np.array(emb.data[0].embedding, dtype="float32")
        sims = cosine_similarities(embeddings, q_vec)
        top = sims.argsort()[-20:][::-1]
        contexts = [metadata[i]["text"] for i in top]
        prompt   = "Use these passages:\n\n" + "\n---\n".join(contexts)
        prompt  += f"\n\nQuestion: {question}\nAnswer:"
        chat     = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user",  "content":prompt}
            ]
        )
        raw     = chat.choices[0].message.content
        answer  = format_response(remove_bullets(raw))

        # 7) Fallback URL + label
        if not relevant_url and top.size:
            relevant_url = metadata[top[0]].get("url")
        link_label = URL_LABELS.get(relevant_url)

        return jsonify(answer=answer, url=relevant_url, link_label=link_label), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify(error=str(e)), 500

# ─── Run the app ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
