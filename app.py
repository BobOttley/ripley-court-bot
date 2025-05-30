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
    "home":                                "https://www.ripleycourt.co.uk/",
    "welcome from the headmaster":         "https://www.ripleycourt.co.uk/19/welcome-from-the-headmaster",
    "introduction to ripley court":        "https://www.ripleycourt.co.uk/20/an-introduction-to-ripley-court-school",
    "what our parents say about us":       "https://www.ripleycourt.co.uk/972/what-our-parents-say-about-us",
    "our campus":                          "https://www.ripleycourt.co.uk/734/our-campus",
    "aims ethos and values":              "https://www.ripleycourt.co.uk/21/aims-ethos-and-values",
    "history of the school":               "https://www.ripleycourt.co.uk/22/history-of-the-school",
    "reeds school":                        "https://www.ripleycourt.co.uk/17/reeds-school",
    "results and destinations":            "https://www.ripleycourt.co.uk/24/results-and-destination-schools",
    "inspection reports":                  "https://www.ripleycourt.co.uk/25/inspection-reports",
    "staff members":                       "https://www.ripleycourt.co.uk/130/staff-members",
    "parent teacher association":          "https://www.ripleycourt.co.uk/74/ripley-court-parent-teacher-association-rcpta",
    "alumni":                              "https://www.ripleycourt.co.uk/136/alumni",
    "location":                            "https://www.ripleycourt.co.uk/29/ripley-court-location",
    "policies":                            "https://www.ripleycourt.co.uk/73/policies/category/21/school-policies",
    "term dates":                          "https://www.ripleycourt.co.uk/63/term-dates",
    "calendar":                            "https://www.ripleycourt.co.uk/64/calendar",
    "lunch":                               "https://www.ripleycourt.co.uk/72/menus",
    "sport":                               "https://www.ripleycourt.co.uk/38/sport",
    "uniform":                             "https://www.ripleycourt.co.uk/75/uniform",
    "prospectus":                          "https://www.ripleycourt.co.uk/46/prospectus",
    "enquire":                             "https://www.ripleycourt.co.uk/47/enquire",
    "registration deadlines":              "https://www.ripleycourt.co.uk/48/begin-the-registration-process",
    "fees":                                "https://www.ripleycourt.co.uk/50/fees",
    "scholarships and bursaries":          "https://www.ripleycourt.co.uk/625/scholarships-and-bursaries",
    "little twigs stay play":              "https://www.ripleycourt.co.uk/54/little-twigs-stay-play",
    "campus map":                          "https://ripleycourtsh.s3.amazonaws.com/uploads/document/CampusMap.jpg?ts=1615307855162",
    "nursery":                             "https://www.ripleycourt.co.uk/32/nursery",
    "head of early years and ks1":         "https://www.ripleycourt.co.uk/1425/head-of-early-years-and-ks1",
    "upper court prep":                    "https://www.ripleycourt.co.uk/34/upper-court-prep",
    "prep school 1":                       "https://www.ripleycourt.co.uk/901/prep-school-1",
    "curriculum":                          "https://www.ripleycourt.co.uk/36/curriculum",
    "art":                                 "https://www.ripleycourt.co.uk/221/academics/subject/16/art",
    "computing":                           "https://www.ripleycourt.co.uk/221/academics/subject/8/computing",
    "drama":                               "https://www.ripleycourt.co.uk/221/academics/subject/17/drama",
    "english":                             "https://www.ripleycourt.co.uk/221/academics/subject/1/english",
    "food technology":                    "https://www.ripleycourt.co.uk/221/academics/subject/15/food-technology",
    "french modern foreign languages":     "https://www.ripleycourt.co.uk/221/academics/subject/10/french-modern-foreign-languages",
    "geography":                           "https://www.ripleycourt.co.uk/221/academics/subject/5/geography",
    "history":                             "https://www.ripleycourt.co.uk/221/academics/subject/4/history",
    "learning support":                    "https://www.ripleycourt.co.uk/221/academics/subject/12/learning-support",
    "mathematics":                         "https://www.ripleycourt.co.uk/221/academics/subject/2/mathematics",
    "music":                               "https://www.ripleycourt.co.uk/221/academics/subject/11/music",
    "philosophy":                          "https://www.ripleycourt.co.uk/221/academics/subject/7/philosophy",
    "pshee":                               "https://www.ripleycourt.co.uk/221/academics/subject/13/pshee",
    "religious studies":                   "https://www.ripleycourt.co.uk/221/academics/subject/6/religious-studies",
    "science":                             "https://www.ripleycourt.co.uk/221/academics/subject/3/science",
    "sport pe":                            "https://www.ripleycourt.co.uk/221/academics/subject/14/sport-pe",
    "verbal non-verbal reasoning":         "https://www.ripleycourt.co.uk/221/academics/subject/9/verbal-non-verbal-reasoning",
    "visit us":                            "https://www.ripleycourt.co.uk/44/visit-us",
    "open mornings and visits":            "https://www.ripleycourt.co.uk/45/open-mornings-and-visits",
    "latest news":                         "https://www.ripleycourt.co.uk/55/latest-news",
    "newsletters":                         "https://www.ripleycourt.co.uk/56/newsletters",
    "schoolbase logon":                    "https://schoolbase.online/Logon?DName=Reeds",
    "shop":                                "https://ripleycourtshop.co.uk/",
    "uniform list":                        "https://ripleycourtshop.co.uk/pages/uniform-list",
    "shop contact":                        "https://ripleycourtshop.co.uk/pages/contact",
    "boy nursery transition":              "https://ripleycourtshop.co.uk/collections/boy-nursery-transition",
    "boy reception year 2":                "https://ripleycourtshop.co.uk/collections/boy-reception-year-2",
    "boy year 3-6":                        "https://ripleycourtshop.co.uk/collections/boy-year-3-6",
    "girl nursery transition":             "https://ripleycourtshop.co.uk/collections/girl-nursery-transition",
    "girl reception year 2":               "https://ripleycourtshop.co.uk/collections/girl-reception-year-2",
    "girl year 3-6":                       "https://ripleycourtshop.co.uk/collections/girl-year-3-6",
    "current vacancies":                   "https://www.ripleycourt.co.uk/16/current-vacancies",
}

# ─── Human-readable labels ─────────────────────────────────────────────────
URL_LABELS = {
    PAGE_LINKS["home"]:                               "Home",
    PAGE_LINKS["welcome from the headmaster"]:        "Welcome from the Headmaster",
    PAGE_LINKS["introduction to ripley court"]:       "Introduction to Ripley Court",
    PAGE_LINKS["what our parents say about us"]:      "What Our Parents Say",
    PAGE_LINKS["our campus"]:                         "Our Campus",
    PAGE_LINKS["aims ethos and values"]:              "Aims, Ethos & Values",
    PAGE_LINKS["history of the school"]:              "History of the School",
    PAGE_LINKS["reeds school"]:                       "Reed’s School",
    PAGE_LINKS["results and destinations"]:           "Results & Destinations",
    PAGE_LINKS["inspection reports"]:                 "Inspection Reports",
    PAGE_LINKS["staff members"]:                      "Staff Members",
    PAGE_LINKS["parent teacher association"]:         "Parent-Teacher Association",
    PAGE_LINKS["alumni"]:                             "Alumni",
    PAGE_LINKS["location"]:                           "Location",
    PAGE_LINKS["policies"]:                           "School Policies",
    PAGE_LINKS["term dates"]:                         "Term Dates",
    PAGE_LINKS["calendar"]:                           "Calendar",
    PAGE_LINKS["lunch"]:                              "Lunch Menus",
    PAGE_LINKS["sport"]:                              "Sport",
    PAGE_LINKS["uniform"]:                            "Uniform",
    PAGE_LINKS["prospectus"]:                         "Prospectus",
    PAGE_LINKS["enquire"]:                            "Enquire",
    PAGE_LINKS["registration deadlines"]:             "Begin Registration",
    PAGE_LINKS["fees"]:                               "Fees",
    PAGE_LINKS["scholarships and bursaries"]:         "Scholarships & Bursaries",
    PAGE_LINKS["little twigs stay play"]:             "Little Twigs Stay & Play",
    PAGE_LINKS["campus map"]:                         "Campus Map",
    PAGE_LINKS["nursery"]:                            "Nursery",
    PAGE_LINKS["head of early years and ks1"]:        "Head of Early Years & KS1",
    PAGE_LINKS["upper court prep"]:                   "Upper Court Prep",
    PAGE_LINKS["prep school 1"]:                      "Prep School 1",
    PAGE_LINKS["curriculum"]:                         "Curriculum",
    PAGE_LINKS["art"]:                                "Art",
    PAGE_LINKS["computing"]:                          "Computing",
    PAGE_LINKS["drama"]:                              "Drama",
    PAGE_LINKS["english"]:                            "English",
    PAGE_LINKS["food technology"]:                    "Food Technology",
    PAGE_LINKS["french modern foreign languages"]:    "French (MFL)",
    PAGE_LINKS["geography"]:                          "Geography",
    PAGE_LINKS["history"]:                            "History",
    PAGE_LINKS["learning support"]:                   "Learning Support",
    PAGE_LINKS["mathematics"]:                        "Mathematics",
    PAGE_LINKS["music"]:                              "Music",
    PAGE_LINKS["philosophy"]:                         "Philosophy",
    PAGE_LINKS["pshee"]:                              "PSHEE",
    PAGE_LINKS["religious studies"]:                  "Religious Studies",
    PAGE_LINKS["science"]:                            "Science",
    PAGE_LINKS["sport pe"]:                           "Sport & P.E.",
    PAGE_LINKS["verbal non-verbal reasoning"]:        "Verbal & Non-Verbal Reasoning",
    PAGE_LINKS["visit us"]:                           "Visit Us",
    PAGE_LINKS["open mornings and visits"]:           "Open Mornings & Visits",
    PAGE_LINKS["latest news"]:                        "Latest News",
    PAGE_LINKS["newsletters"]:                        "Newsletters",
    PAGE_LINKS["schoolbase logon"]:                   "SchoolBase Logon",
    PAGE_LINKS["shop"]:                               "School Shop",
    PAGE_LINKS["uniform list"]:                       "Uniform List",
    PAGE_LINKS["shop contact"]:                       "Shop Contact",
    PAGE_LINKS["boy nursery transition"]:             "Boy Nursery Transition",
    PAGE_LINKS["boy reception year 2"]:               "Boy Reception–Year 2",
    PAGE_LINKS["boy year 3-6"]:                       "Boy Years 3–6",
    PAGE_LINKS["girl nursery transition"]:            "Girl Nursery Transition",
    PAGE_LINKS["girl reception year 2"]:              "Girl Reception–Year 2",
    PAGE_LINKS["girl year 3-6"]:                      "Girl Years 3–6",
    PAGE_LINKS["current vacancies"]:                  "Current Vacancies",
}

# ─── Headmaster config ───────────────────────────────────────────────────────
# update HEADMASTER_NAME (or .env) only here when the post changes
HEADMASTER_NAME     = os.getenv("HEADMASTER_NAME", "Mr Gavin Ryan")
HEADMASTER_PAGE_KEY = "welcome from the headmaster"

# safe lookup with a clear error if you typo the key
HEADMASTER_URL = PAGE_LINKS.get(HEADMASTER_PAGE_KEY)
if not HEADMASTER_URL:
    raise RuntimeError(f"No PAGE_LINKS entry for '{HEADMASTER_PAGE_KEY}'")

HEADMASTER_LABEL = URL_LABELS.get(HEADMASTER_URL, "Welcome from the Headmaster")

HEADMASTER_TEXT = (
    f"The Headmaster is {HEADMASTER_NAME}. "
    "For more about his vision and background, "
    "please visit our Welcome from the Headmaster page."
)

# ─── Ripley Court Static QAs ─────────────────────────────────────────────────
STATIC_QAS = {
    # ─── Enquiry & Prospectus ────────────────────────────────────────────────
    "enquiry": (
        "Please complete our enquiry form and we will tailor a prospectus exactly for you and your family.",
        PAGE_LINKS["enquire"],
        URL_LABELS[PAGE_LINKS["enquire"]]
    ),
    "ask about enquiry": (
        "Please complete our enquiry form and we will tailor a prospectus exactly for you and your family.",
        PAGE_LINKS["enquire"],
        URL_LABELS[PAGE_LINKS["enquire"]]
    ),
    "prospectus": (
        "Download our full prospectus below to learn more about life at Ripley Court School.",
        PAGE_LINKS["prospectus"],
        URL_LABELS[PAGE_LINKS["prospectus"]]
    ),
    "download prospectus": (
        "Download our full prospectus below to learn more about life at Ripley Court School.",
        PAGE_LINKS["prospectus"],
        URL_LABELS[PAGE_LINKS["prospectus"]]
    ),
    "view prospectus": (
        "Download our full prospectus below to learn more about life at Ripley Court School.",
        PAGE_LINKS["prospectus"],
        URL_LABELS[PAGE_LINKS["prospectus"]]
    ),

    # ─── Fees ────────────────────────────────────────────────────────────────
    "fees": (
        "Our current fees and payment options are detailed on the Fees page.",
        PAGE_LINKS["fees"],
        URL_LABELS[PAGE_LINKS["fees"]]
    ),

    # ─── Uniform ────────────────────────────────────────────────────────────
    "uniform": (
        "Full uniform requirements are listed on our Uniform page—including sports kit and second-hand options.",
        PAGE_LINKS["uniform"],
        URL_LABELS[PAGE_LINKS["uniform"]]
    ),

    # ─── School Lunches & Menus ────────────────────────────────────────────
    "lunch": (
        "You can download our current school lunch menus (including dietary options) here:",
        PAGE_LINKS["lunch"],
        URL_LABELS[PAGE_LINKS["lunch"]]
    ),

    # ─── Term Dates & Holidays ─────────────────────────────────────────────
    "term dates": (
        "Our published term dates, half-terms and holiday dates can be found here:",
        PAGE_LINKS["term dates"],
        URL_LABELS[PAGE_LINKS["term dates"]]
    ),

    # ─── Registration & Admissions ────────────────────────────────────────
    "registration deadlines": (
        "Key registration deadlines for Nursery, Reception and Year 3 (and rolling entry) are on our Registration page.",
        PAGE_LINKS["registration deadlines"],
        URL_LABELS[PAGE_LINKS["registration deadlines"]]
    ),

    # ─── Open Events & Visits ──────────────────────────────────────────────
    "open events": (
        "Join us for Open Mornings or book a private visit to experience Ripley Court firsthand.",
        PAGE_LINKS["open mornings and visits"],                  
        URL_LABELS[PAGE_LINKS["open mornings and visits"]]
    ),
    "open mornings": (
        "Join us for Open Mornings or book a private visit to experience Ripley Court firsthand.",
        PAGE_LINKS["open mornings and visits"],                  
        URL_LABELS[PAGE_LINKS["open mornings and visits"]]
    ),
    "visit us": (
        "Arrange a personal tour to experience Ripley Court’s welcoming community and outstanding pastoral care.",
        PAGE_LINKS["visit us"],                                  
        URL_LABELS[PAGE_LINKS["visit us"]]
    ),

    # ─── Scholarships & Bursaries ──────────────────────────────────────────
    "scholarships": (
        "We offer a range of subject scholarships (Academic, Sport, Music, Art) and means-tested bursaries. See details here.",
        PAGE_LINKS["scholarships and bursaries"],               
        URL_LABELS[PAGE_LINKS["scholarships and bursaries"]]
    ),
    "bursaries": (
        "We offer a range of subject scholarships (Academic, Sport, Music, Art) and means-tested bursaries. See details here.",
        PAGE_LINKS["scholarships and bursaries"],               
        URL_LABELS[PAGE_LINKS["scholarships and bursaries"]]
    ),

    # ─── Contact & Staff ──────────────────────────────────────────────────
    "contact": (
        "For general or admissions enquiries, email registrar@ripleycourt.co.uk or call 01483 225 217.",
        PAGE_LINKS["enquire"],
        URL_LABELS[PAGE_LINKS["enquire"]]
    ),
    "staff": (
        "Meet our dedicated teaching and support staff who guide every pupil’s academic and personal growth.",
        PAGE_LINKS["staff members"],
        URL_LABELS[PAGE_LINKS["staff members"]]
    ),

    # ─── Campus & Ethos ───────────────────────────────────────────────────
    "campus": (
        "Explore our 19-acre Surrey campus with playing fields, forest school, indoor pool and specialist labs.",
        PAGE_LINKS["our campus"],
        URL_LABELS[PAGE_LINKS["our campus"]]
    ),
    "ethos": (
        "Our values—Perseverance, Aspiration, Curiosity and Kindness—underpin everything we do.",
        PAGE_LINKS["aims ethos and values"],
        URL_LABELS[PAGE_LINKS["aims ethos and values"]]
    ),

    # ─── History & Outcomes ────────────────────────────────────────────────
    "history": (
        "Founded in 1886, Ripley Court has grown from a boarding school into today’s thriving prep and joined Reed’s foundation in 2019.",
        PAGE_LINKS["history of the school"],
        URL_LABELS[PAGE_LINKS["history of the school"]]
    ),
    "results": (
        "View recent Year 6 destinations and scholarship outcomes on our Results & Destinations page.",
        PAGE_LINKS["results and destinations"],
        URL_LABELS[PAGE_LINKS["results and destinations"]]
    ),
    "destinations": (
        "View recent Year 6 destinations and scholarship outcomes on our Results & Destinations page.",
        PAGE_LINKS["results and destinations"],
        URL_LABELS[PAGE_LINKS["results and destinations"]]
    ),

    # ─── School Divisions & Curriculum ────────────────────────────────────
    "nursery": (
        "Our Nursery provides a rich, multi-sensory EYFS environment for children from age 3, with wrap-around care from 7:30 am – 6 pm.",
        PAGE_LINKS["nursery"],
        URL_LABELS[PAGE_LINKS["nursery"]]
    ),
    "upper court prep": (
        "Upper Court Prep (Years 3–6) offers specialist teaching, enrichment and 11+ preparation in small, supportive classes.",
        PAGE_LINKS["upper court prep"],
        URL_LABELS[PAGE_LINKS["upper court prep"]]
    ),
    "prep school": (
        "The Prep School (Years 3–6) combines specialist teachers, a broad curriculum and pastoral care to prepare pupils for senior school.",
        PAGE_LINKS["prep school 1"],
        URL_LABELS[PAGE_LINKS["prep school 1"]]
    ),
    "curriculum": (
        "Our broad curriculum spans core subjects, the arts, languages and STEM—enriched by trips, forest school and practical labs.",
        PAGE_LINKS["curriculum"],
        URL_LABELS[PAGE_LINKS["curriculum"]]
    ),

    # ─── News, Newsletters & Careers ───────────────────────────────────────
    "latest news": (
        "Stay up to date with the latest Ripley Court news and announcements.",
        PAGE_LINKS["latest news"],
        URL_LABELS[PAGE_LINKS["latest news"]]
    ),
    "newsletters": (
        "Access all our past newsletters for highlights on school life and upcoming events.",
        PAGE_LINKS["newsletters"],
        URL_LABELS[PAGE_LINKS["newsletters"]]
    ),
    "current vacancies": (
        "Join our vibrant team at Ripley Court—view current teaching and support vacancies here.",
        PAGE_LINKS["current vacancies"],
        URL_LABELS[PAGE_LINKS["current vacancies"]]
    ),

    # ─── Login & Alumni ───────────────────────────────────────────────────
    "logon": (
        "Need to reset your password? Enter your email on the SchoolBase portal to receive your login details and reset link.",
        PAGE_LINKS["schoolbase logon"],
        URL_LABELS[PAGE_LINKS["schoolbase logon"]]
    ),
    "alumni": (
        "Join our Alumnae network for mentorship, events and lifelong connections with Ripley Court past pupils.",
        PAGE_LINKS["alumni"],
        URL_LABELS[PAGE_LINKS["alumni"]]
    ),

    # ─── Shop & Uniform Supplier ────────────────────────────────────────────
    "shop": (
        "Visit the Ripley Court Shop for branded gifts, accessories and uniform items.",
        PAGE_LINKS["shop"],
        URL_LABELS[PAGE_LINKS["shop"]]
    ),
    "uniform list": (
        "Browse our full uniform list—order essential items for each year group online.",
        PAGE_LINKS["uniform list"],
        URL_LABELS[PAGE_LINKS["uniform list"]]
    ),
    "shop contact": (
        "For uniform enquiries or orders, email shop@reeds.surrey.sch.uk or call 01932 869065.",
        PAGE_LINKS["shop contact"],
        URL_LABELS[PAGE_LINKS["shop contact"]]
    ),

    # ─── Shop Categories (optional) ─────────────────────────────────────────
    "boy nursery transition": (
        "Shop the boy Nursery & Transition uniform collection: essentials for our youngest pupils.",
        PAGE_LINKS["boy nursery transition"],
        URL_LABELS[PAGE_LINKS["boy nursery transition"]]
    ),
    "boy reception year 2": (
        "Find boys’ Reception–Year 2 uniform and PE kit in our online shop.",
        PAGE_LINKS["boy reception year 2"],
        URL_LABELS[PAGE_LINKS["boy reception year 2"]]
    ),
    "boy year 3-6": (
        "Shop boys’ Years 3–6 uniform: blazers, jumpers, sports kit and more.",
        PAGE_LINKS["boy year 3-6"],
        URL_LABELS[PAGE_LINKS["boy year 3-6"]]
    ),
    "girl nursery transition": (
        "Browse the girl Nursery & Transition uniform collection for our youngest ladies.",
        PAGE_LINKS["girl nursery transition"],
        URL_LABELS[PAGE_LINKS["girl nursery transition"]]
    ),
    "girl reception year 2": (
        "Shop girls’ Reception–Year 2 uniform: gingham dresses, cardigans and essentials.",
        PAGE_LINKS["girl reception year 2"],
        URL_LABELS[PAGE_LINKS["girl reception year 2"]]
    ),
    "girl year 3-6": (
        "Find girls’ Years 3–6 uniform including blazers, skirts, PE kit and accessories.",
        PAGE_LINKS["girl year 3-6"],
        URL_LABELS[PAGE_LINKS["girl year 3-6"]]
    ),
    "sixth form": (
        "We educate pupils from Nursery through Year 6, preparing them academically and pastorally "
        "for the transition to senior schools.",
        PAGE_LINKS["results and destinations"],
        URL_LABELS[PAGE_LINKS["results and destinations"]]
    ),
    "what subjects do you offer"] = (
    "We offer a broad curriculum across Art, Computing, Drama, English, Food Technology, French, Geography, History, Learning Support, Mathematics, Music, Philosophy, PSHEE, Religious Studies, Science, Sport & PE,Verbal & Non-Verbal Reasoning and more.",
    PAGE_LINKS["curriculum"],
    URL_LABELS[PAGE_LINKS["curriculum"]]
)
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

        # 5) Keyword → URL (only match full keys longer than 6 chars)
        relevant_url = None
        for k, u in PAGE_LINKS.items():
           if len(k) > 6 and k in key:
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
