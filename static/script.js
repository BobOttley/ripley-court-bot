// /static/script.js
console.log("🚀 script.js loaded");

document.addEventListener("DOMContentLoaded", () => {
  // ─── UI refs ─────────────────────────────────────────────────
  const toggle       = document.getElementById("penai-toggle");
  const chatbox      = document.getElementById("penai-chatbox");
  const msgs         = document.getElementById("penai-messages");
  const input        = document.getElementById("penai-input");
  const sendBtn      = document.getElementById("penai-send-btn");
  const header       = document.getElementById("penai-header");
  const minimizeBtn  = document.getElementById("penai-minimize-btn");

  // ─── Chat endpoint ─────────────────────────────────────────────
  const ASK_URL = "http://127.0.0.1:5000/ask";

  // ─── State ────────────────────────────────────────────────────
  let chatHistory       = [];
  let welcomed          = false;
  let exitIntentShown   = false;
  const usedQueries     = new Set();
  let thinkingDiv       = null;
  let currentController = null;

  // ─── Quick-replies grouped by category ────────────────────────
  const quickByCat = {
    admissions: [
      { label: "Enquire now",            query: "enquire" },
      { label: "Download prospectus",    query: "prospectus" },
      { label: "Fees",                   query: "fees" },
      { label: "Registration deadlines", query: "registration deadlines" }
    ],
    lunch: [
      { label: "Lunch menu",             query: "lunch" }
    ],
    calendar: [
      { label: "Term dates",             query: "term dates" },
      { label: "Open events",            query: "open events" }
    ],
    uniform: [
      { label: "Uniform info",           query: "uniform" }
    ],
    scholarships: [
      { label: "Bursaries & scholarships", query: "scholarships and bursaries" }
    ],
    contact: [
      { label: "Contact us",             query: "contact" }
    ],
    academics: [
      { label: "Academic life",          query: "academic life" },
      { label: "Subjects offered",       query: "subjects" },
      { label: "Sixth Form",             query: "sixth form" }
    ],
    extracurricular: [
      { label: "Co-curricular",          query: "co-curricular" },
      { label: "Sport",                  query: "sport" },
      { label: "Faith Life",             query: "faith life" }
    ],
    policies: [
      { label: "Policies",               query: "policies" },
      { label: "Safeguarding",           query: "safeguarding" }
    ]
  };

  // ─── Helpers ─────────────────────────────────────────────────
  function renderParagraphs(text) {
    return text
      .split(/\n{2,}/)
      .map(p => `<p>${p.trim()}</p>`)
      .join("");
  }

  function detectCategory(text) {
    const t = text.toLowerCase();
    if (/(register|registration|admission|fee|prospectus)/.test(t)) return "admissions";
    if (/(lunch|dietary|menu|meal)/.test(t))             return "lunch";
    if (/(term|holiday|calendar|event)/.test(t))         return "calendar";
    if (/(uniform|dress code)/.test(t))                  return "uniform";
    if (/(bursary|scholarship)/.test(t))                 return "scholarships";
    if (/(contact|email|phone|telephone)/.test(t))       return "contact";
    if (/(academic|subject|learning)/.test(t))           return "academics";
    if (/(sport|co-curricular|activity)/.test(t))        return "extracurricular";
    if (/(policy|policies|safeguarding)/.test(t))        return "policies";
    return "admissions";
  }

  function saveHistory(type, text) {
    chatHistory.push({ type, text });
  }

  function loadHistory() {
    msgs.innerHTML = "";
    for (const { type, text } of chatHistory) {
      if (type === "user") {
        renderUser(text, false);
      } else {
        const cat = detectCategory(text);
        renderBot(text, false, cat, false);
      }
    }
  }

  // ─── Fade out old messages ────────────────────────────────────
  function fadeOldMessages() {
    document
      .querySelectorAll("#penai-messages .penai-message")
      .forEach(el => el.classList.add("faded"));
  }

  // ─── Render user message ──────────────────────────────────────
  function renderUser(text, save = true) {
    fadeOldMessages();
    Object.values(quickByCat).flat().forEach(b => {
      if (b.query.toLowerCase() === text.toLowerCase()) {
        usedQueries.add(b.query);
      }
    });
    const d = document.createElement("div");
    d.className = "penai-message penai-user";
    d.innerHTML = `<strong>Me:</strong> ${text}`;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
    if (save) saveHistory("user", text);
  }

  // ─── Render bot message + quick-replies ───────────────────────
  function renderBot(html, isWelcome = false, category = "admissions", save = true) {
    fadeOldMessages();
    const d = document.createElement("div");
    d.className = "penai-message penai-bot";
    d.innerHTML = `<strong><span class="penai-prefix">PEN.ai:</span></strong> ${html}`;
    msgs.appendChild(d);

    // build quick-reply buttons
    const cats   = Object.keys(quickByCat);
    const inCat  = quickByCat[category]   || quickByCat.admissions;
    const outCat = cats.filter(c => c !== category).flatMap(c => quickByCat[c]);

    const freshIn  = inCat.filter(b => !usedQueries.has(b.query));
    const seenIn   = inCat.filter(b =>  usedQueries.has(b.query));
    const freshOut = outCat.filter(b => !usedQueries.has(b.query));
    const seenOut  = outCat.filter(b =>  usedQueries.has(b.query));

    const finalBtns = freshIn.concat(freshOut, seenIn, seenOut).slice(0,3);

    const wrp = document.createElement("div");
    wrp.className = "quick-replies";
    wrp.innerHTML = finalBtns
      .map(b => `<button class="quick-reply" data-query="${b.query}">${b.label}</button>`)
      .join("");
    d.appendChild(wrp);

    msgs.scrollTop = msgs.scrollHeight;
    if (save) saveHistory("bot", html);
  }

  // ─── Thinking indicator ──────────────────────────────────────
  function showThinking() {
    removeThinking();
    thinkingDiv = document.createElement("div");
    thinkingDiv.className = "penai-message penai-bot";
    thinkingDiv.innerHTML =
      `<strong><span class="penai-prefix">PEN.ai:</span></strong> <em>PEN.ai is thinking…</em>`;
    msgs.appendChild(thinkingDiv);
    msgs.scrollTop = msgs.scrollHeight;
  }
  function removeThinking() {
    if (thinkingDiv && msgs.contains(thinkingDiv)) {
      msgs.removeChild(thinkingDiv);
      thinkingDiv = null;
    }
  }

  // ─── Send a question ──────────────────────────────────────────
  async function sendQuestion(question, isWelcome = false) {
    if (currentController) currentController.abort();
    currentController = new AbortController();

    if (!isWelcome) {
      renderUser(question);
      showThinking();
    }

    try {
      const res = await fetch(ASK_URL, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ question }),
        signal:  currentController.signal
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      removeThinking();
      usedQueries.add(question);

      // sanitize URLs in answer
      let linked = data.answer.replace(
        /(https?:\/\/[^\s]+)/g,
        '<a href="$1" target="_blank">$1</a>'
      );

      // strip footer, render core paragraphs
      const footer = "Anything else I can help you with today?";
      let core   = linked.replace(footer, "").trim();
      let html   = renderParagraphs(core);

      // append link label
      if (data.url) {
        const lbl = data.link_label || "More details";
        html += `<p><a href="${data.url}" target="_blank">${lbl}</a></p>`;
      }
      // re-add footer
      html += `<p>${footer}</p>`;

      const cat = detectCategory(data.answer);
      renderBot(html, isWelcome, cat);

    } catch (err) {
      if (err.name === "AbortError") {
        removeThinking();
      } else {
        removeThinking();
        renderBot(`Sorry, I couldn't process your request. (${err.message})`,
                  false, "admissions");
      }
    } finally {
      currentController = null;
    }
  }

  // ─── Toggle open/close ───────────────────────────────────────
  toggle.addEventListener("click", () => {
    const isOpen = chatbox.style.display === "flex";
    if (isOpen) {
      if (currentController) currentController.abort();
      chatbox.style.display = "none";
    } else {
      chatbox.style.display = "flex";
      if (!welcomed) {
        sendQuestion("__welcome__", true);
        welcomed = true;
      } else {
        loadHistory();
      }
    }
  });

  // ─── Minimize button ─────────────────────────────────────────
  minimizeBtn.addEventListener("click", () => {
    if (currentController) currentController.abort();
    chatbox.style.display = "none";
  });

  // ─── Input & send handlers ───────────────────────────────────
  sendBtn.addEventListener("click", () => {
    const msg = input.value.trim();
    if (msg) {
      input.value = "";
      sendQuestion(msg, false);
    }
  });
  input.addEventListener("keypress", e => {
    if (e.key === "Enter" && input.value.trim()) {
      const msg = input.value.trim();
      input.value = "";
      sendQuestion(msg, false);
    }
  });

  // ─── Quick-reply clicks ──────────────────────────────────────
  document.addEventListener("click", e => {
    if (e.target.classList.contains("quick-reply")) {
      const q = e.target.getAttribute("data-query");
      usedQueries.add(q);
      sendQuestion(q, false);
    }
  });

  // ─── Draggable chatbox ───────────────────────────────────────
  let dragging = false, offsetX, offsetY;
  header.addEventListener("mousedown", e => {
    dragging = true;
    offsetX = e.clientX - chatbox.offsetLeft;
    offsetY = e.clientY - chatbox.offsetTop;
  });
  document.addEventListener("mousemove", e => {
    if (dragging) {
      chatbox.style.left = `${e.clientX - offsetX}px`;
      chatbox.style.top  = `${e.clientY - offsetY}px`;
    }
  });
  document.addEventListener("mouseup", () => {
    dragging = false;
  });

  // ─── Exit-intent nudge ───────────────────────────────────────
  function showExitNudge() {
    if (exitIntentShown) return;
    exitIntentShown = true;
    if (chatbox.style.display !== "flex") toggle.click();
    setTimeout(() => {
      const prompt = "Before you go, would you like to request your personalised prospectus?";
      const linkHtml = '<p><a href="https://www.ripleycourt.co.uk/47/enquire" target="_blank">Request your prospectus here</a></p>';
      renderBot(`<em>${prompt}</em>${linkHtml}`, false, detectCategory(prompt));
    }, 200);
  }
  document.addEventListener("mouseout", e => {
    if (!exitIntentShown
        && chatbox.style.display === "none"
        && e.clientY <= 0
        && !e.relatedTarget) {
      showExitNudge();
    }
  });

  // ─── Auto-open after 20s if never opened ─────────────────────
  setTimeout(() => {
    if (!welcomed && chatbox.style.display === "none") {
      toggle.click();
    }
  }, 20000);

  // ─── Start hidden ────────────────────────────────────────────
  chatbox.style.display = "none";
});
