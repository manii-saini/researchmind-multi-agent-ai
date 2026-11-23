import os
import re
import time
from urllib.parse import urlparse

import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "").strip()
REQUEST_TIMEOUT = int(os.getenv("LOCAL_LLM_TIMEOUT", "90"))

_MODEL_CACHE = {"checked": 0.0, "model": None}
_CACHE_SECONDS = 20

STOPWORDS = {
    "about", "after", "against", "based", "being", "between", "breakthrough",
    "current", "detail", "details", "develop", "development", "does", "from",
    "have", "information", "into", "latest", "recent", "reliable", "report",
    "research", "source", "sources", "that", "their", "there", "these", "this",
    "those", "through", "topic", "using", "what", "when", "where", "which",
    "while", "with", "would", "your",
}

BAD_MARKERS = (
    "jump to content", "cookie", "privacy policy", "terms of use", "edit links",
    "main menu", "navigation", "subscribe", "all rights reserved", "sign up",
    "log in", "sign in", "languages", "search this", "view source", "random article",
    "skip to main", "skip to content", "opens in a new tab", "read the blog",
    "watch the video", "advertisement", "open access", "saved research",
    "view all journals", "latest research and news", "functions to support",
    "use cookies", "your browser", "compatibility mode", "check your browser",
    "article open access", "###", "  # ", "import qiskit", "nighthawk runs",
)

VAGUE_WORDS = ("various", "several", "many", "some", "many experts", "a number of", "good", "nice", "things")

REPORT_PROMPT = """You are a senior research analyst. Write a detailed, evidence-based report using ONLY the research material supplied below.

Topic: {topic}

Research Gathered:
{research}

Requirements:
- Write at least 700 words.
- Use this exact structure:
# {topic}
## Executive Summary
## Introduction
## Key Findings
## Themes and Analysis
## Limitations and Caveats
## Conclusion
## Sources
- Key Findings must contain at least 5 numbered points, each with 3 or more sentences.
- Cite every factual claim inline as [publisher](URL), using only URLs present in the research.
- Use the scraped page text, not only the search snippets.
- Never invent statistics, dates, quotations or sources. If evidence is weak, say so.
- Be specific, balanced and professional."""

FEEDBACK_PROMPT = """You are a strict research reviewer. Score the report below out of 10 using a deduction rubric: start at 10 and subtract points for short length, fewer than 5 findings, fewer than 3 unique sources, missing sections, unsupported claims, no quantitative or date evidence, vague language, and repetition.

A score of 10 is reserved for exceptional work (900+ words, 7+ sourced findings, 8+ unique sources, quantitative evidence, limitations section, no repetition). Do not give 10 by default. Typical solid reports score 7-8.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- specific point 1
- specific point 2

Areas to Improve:
- specific point 1
- specific point 2

One line verdict:
one short sentence"""


def local_model():
    now = time.time()
    if now - _MODEL_CACHE["checked"] < _CACHE_SECONDS:
        return _MODEL_CACHE["model"]
    _MODEL_CACHE["checked"] = now
    _MODEL_CACHE["model"] = None
    try:
        response = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2)
        response.raise_for_status()
        models = [
            item.get("name")
            for item in response.json().get("models", [])
            if item.get("name")
        ]
    except Exception:
        return None
    if OLLAMA_MODEL:
        if OLLAMA_MODEL in models:
            _MODEL_CACHE["model"] = OLLAMA_MODEL
        return _MODEL_CACHE["model"]
    if models:
        _MODEL_CACHE["model"] = models[0]
    return _MODEL_CACHE["model"]


def backend_info():
    model = local_model()
    if model:
        return f"Local Ollama model active: {model} (no API key required)"
    return "Offline analyzer active (no API key, no network required)"


def generate(prompt, task="report", topic="", research="", report=""):
    model = local_model()
    if model:
        try:
            response = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 1600},
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            text = (response.json().get("response") or "").strip()
            if text and len(text.split()) >= 150 and "Key Findings" in text:
                return text
        except Exception:
            pass
    if task == "feedback":
        return build_feedback(report or prompt)
    return build_report(topic, research or prompt)


def _field(block, label):
    match = re.search(
        rf"^[ \t]*{re.escape(label)}[ \t]*(.+)$", block, re.IGNORECASE | re.MULTILINE
    )
    return match.group(1).strip() if match else ""


def _clean(text, limit=280):
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if boundary > 80:
        cut = cut[: boundary + 1]
    return cut.rstrip() + "…"


def _sanitize_snippet(text):
    text = re.sub(r"^#{1,6}\s*Description\s*", "", text or "", flags=re.IGNORECASE)
    text = re.sub(r"[\d,]+\s+views\s+Posted:\s*[\d-]+\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\[[^\]]+\]\s*", "", text)
    return text.strip(" -–—")


def _tidy(sentence):
    text = _clean(sentence, 380).strip(" -–—")
    return text[:1].upper() + text[1:] if text else text


def _domain(url):
    return (urlparse(url).netloc or "source").replace("www.", "").lower()


def _normalize_url(url):
    return (url or "").strip().rstrip("),.;'\"]>\u200b")


def _parse_research(research):
    text = research or ""
    sources = {}
    order = []

    def ensure(url, title=""):
        url = _normalize_url(url)
        if not url.lower().startswith("http"):
            return None
        if url not in sources:
            sources[url] = {
                "title": title or _domain(url),
                "url": url,
                "snippet": "",
                "body": "",
            }
            order.append(url)
        if title and sources[url]["title"] in (_domain(url), ""):
            sources[url]["title"] = title
        return sources[url]

    for block in re.split(r"\n\s*-{3,}\s*\n", text):
        url = _field(block, "URL:")
        if not url:
            found = re.search(r"https?://[^\s)\]]+", block)
            url = found.group(0) if found else ""
        entry = ensure(url, _field(block, "Title:"))
        if entry:
            snippet = _sanitize_snippet(_field(block, "Snippet:"))
            if len(snippet) >= 40:
                entry["snippet"] = _clean(snippet, 420)

    pattern = r"Source:\s*(https?://[^\s]+)\s*\n+(.+?)(?=\n\s*-{3,}\s*\n|$)"
    for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
        entry = ensure(match.group(1))
        if entry:
            entry["body"] = re.sub(r"\s+", " ", match.group(2)).strip()

    return [sources[url] for url in order]


def _keywords(topic):
    words = re.findall(r"[a-zA-Z]{4,}", topic or "")
    return [word.lower() for word in words if word.lower() not in STOPWORDS][:8]


def _sentences(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if part.strip()]


def _score_sentence(sentence, keywords):
    words = re.findall(r"[A-Za-z0-9%\-]+", sentence)
    if len(words) < 9 or len(words) > 90:
        return 0
    lowered = sentence.lower()
    if any(marker in lowered for marker in BAD_MARKERS):
        return 0
    letters = sum(char.isascii() and char.isalpha() for char in sentence)
    if letters < 0.6 * len(sentence):
        return 0
    if re.search(r"\[\s*\d+\s*\]", sentence) or "://" in sentence:
        return 0
    if sum(1 for word in words if len(word) > 2 and word.isupper()) > 2:
        return 0
    overlap = sum(1 for keyword in keywords if keyword in lowered)
    if overlap == 0 and len(words) < 18:
        return 0
    return overlap + min(len(words), 40) / 40.0


def _findings(sources, keywords):
    results = []
    for source in sources:
        candidates = []
        if source["snippet"]:
            candidates.append((3.0 + len(keywords) / 4.0, _tidy(source["snippet"])))
        for sentence in _sentences(source["body"]):
            score = _score_sentence(sentence, keywords)
            if score:
                candidates.append((score, _tidy(sentence)))
        candidates.sort(key=lambda item: item[0], reverse=True)
        picked = []
        for _, candidate in candidates:
            if any(candidate[:70].lower() == old[:70].lower() for old in picked):
                continue
            picked.append(candidate)
            if len(picked) == 3:
                break
        if not picked:
            picked = [
                f"The available material from {_domain(source['url'])} identifies this source as relevant to the topic, "
                f"but the automated extraction did not yield a clean, quotable passage."
            ]
        results.append(
            {
                "title": source["title"],
                "url": source["url"],
                "text": " ".join(picked),
                "depth": "full page text and search snippet" if source["body"] else "search snippet only",
            }
        )
    return results


def _source_kind(url):
    domain = _domain(url)
    if any(part in domain for part in ("wikipedia", "britannica")):
        return "reference"
    if any(part in domain for part in ("edu", "gov", "nih", "nature", "arxiv", "springer", "sciencedirect", "ieee", "pmc")):
        return "academic or public research"
    return "industry or organizational"


def build_report(topic, research):
    topic = (topic or "this topic").strip()
    sources = _parse_research(research)
    keywords = _keywords(topic)
    scraped_count = sum(1 for source in sources if source["body"])

    if not sources:
        summary = _clean(research, 1200) or "No source material was returned by the research pipeline."
        return (
            f"# {topic}\n\n"
            f"## Executive Summary\n\n"
            f"This report covers **{topic}** using the material the research pipeline was able to collect. "
            f"No verifiable URLs were extracted, so the conclusions below are explicitly provisional.\n\n"
            f"## Introduction\n\n"
            f"The automated search step did not return structured source records for this topic. "
            f"The analysis therefore works directly from the raw research output and flags the "
            f"resulting evidence gap rather than presenting assumptions as facts.\n\n"
            f"## Key Findings\n\n"
            f"1. {summary}\n\n"
            f"2. Because no source URLs were captured, none of the claims above can be independently "
            f"traced to a publisher in this report.\n\n"
            f"3. Re-running the search with narrower wording or connecting a search provider would "
            f"materially improve the reliability of the result.\n\n"
            f"## Themes and Analysis\n\n"
            f"The available material is too thin to support a reliable comparison of competing "
            f"viewpoints. It should be treated as a starting point for targeted follow-up research, "
            f"not as a settled summary.\n\n"
            f"## Limitations and Caveats\n\n"
            f"No source URLs, publication dates or full texts were available. Factual verification "
            f"was therefore not possible.\n\n"
            f"## Conclusion\n\n"
            f"The pipeline produced an overview of {topic}, but the absence of traceable sources "
            f"limits how far the findings can be trusted.\n\n"
            f"## Sources\n\n"
            f"- No verifiable URLs were retrieved in this run.\n"
        )

    findings = _findings(sources, keywords)
    finding_blocks = "\n\n".join(
        f"{index}. **{item['title']}** — {item['text']} "
        f"([{_domain(item['url'])}]({item['url']}))"
        for index, item in enumerate(findings, 1)
    )

    kinds = {}
    for source in sources:
        kinds[_source_kind(source["url"])] = kinds.get(_source_kind(source["url"]), 0) + 1
    kind_text = ", ".join(f"{count} {name}" for name, count in sorted(kinds.items()))

    leading = " ".join(_sentences(findings[0]["text"])[:2]) or findings[0]["text"]
    second = " ".join(_sentences(findings[1]["text"])[:2]) if len(findings) > 1 else ""
    summary_tail = f" {_clean(second, 240)}" if second else ""

    source_lines = "\n".join(
        f"- [{item['title']}]({item['url']}) — {_domain(item['url'])}; "
        f"{'full page text extracted' if item['depth'].startswith('full') else 'search snippet only'}"
        for item in findings
    )

    return (
        f"# {topic}\n\n"
        f"## Executive Summary\n\n"
        f"This report examines **{topic}** across {len(sources)} distinct source(s), of which "
        f"{scraped_count} could be read in full. The evidence base includes {kind_text}. "
        f"The clearest point from the material is that {_clean(leading[0].lower() + leading[1:] if leading else 'the topic is multifaceted', 260)}"
        f"{summary_tail}\n\n"
        f"## Introduction\n\n"
        f"The research pipeline gathered web results for **{topic}**, then opened the most relevant "
        f"pages to extract longer passages. This report synthesizes both layers: short result "
        f"summaries and full-page text. {len(sources)} unique URL(s) were retained after removing "
        f"duplicates, and {scraped_count} page(s) produced readable body text. The aim is to give a "
        f"source-traceable overview rather than a personal opinion, and every finding below points "
        f"back to the publisher it came from.\n\n"
        f"## Key Findings\n\n"
        f"{finding_blocks}\n\n"
        f"## Themes and Analysis\n\n"
        f"Taken together, the {len(sources)} sources form a mixed evidence base: {kind_text}. "
        f"The strongest recurring theme across the extracted passages is that {topic} is best "
        f"understood through several complementary angles rather than a single claim. The "
        f"reference material supplies definitions and context, while the organizational and "
        f"research sources add applied detail. Where the sources disagree or where the extracted "
        f"text is incomplete, the finding above is deliberately conservative. Readers should "
        f"follow the inline links to check wording, dates and figures at the original publisher.\n\n"
        f"## Limitations and Caveats\n\n"
        f"Only {scraped_count} of {len(sources)} source(s) yielded full page text; the remainder "
        f"contribute search snippets only. Automated extraction can pick up navigation text, and "
        f"pages that block automated access may be underrepresented. No claim in this report has "
        f"been fact-checked against a second independent source, and publication dates were not "
        f"consistently available, so recency cannot be guaranteed.\n\n"
        f"## Conclusion\n\n"
        f"The collected sources provide a practical, traceable overview of {topic}. The most useful "
        f"next step is to verify the highlighted claims against the original pages and to add "
        f"peer-reviewed or primary sources where the current evidence base relies mainly on "
        f"summaries. Within those limits, the findings above represent what the gathered material "
        f"actually supports.\n\n"
        f"## Sources\n\n"
        f"{source_lines}\n"
    )


def _metrics(report):
    text = report or ""
    lowered = text.lower()
    headings = re.findall(r"(?m)^#{1,6}\s+(.+)$", text)
    section_names = [heading.strip().lower() for heading in headings]
    required = ("executive summary", "introduction", "key findings", "themes and analysis", "limitations and caveats", "conclusion", "sources")
    findings = re.findall(r"(?m)^\s*\d+[.)]\s+(?:\*\*)?([^\n]+)", text)
    urls = re.findall(r"https?://[^\s)\]]+", text)
    unique_urls = sorted({url.rstrip("),.;") for url in urls})
    numbers = re.findall(r"(?<!\w)(?:\d{4}|\d+(?:\.\d+)?\s?%|\d+\s?(?:years?|months?|studies|people|countries))", lowered)
    vague = [word for word in VAGUE_WORDS if word in lowered]
    sentences = [sentence for sentence in _sentences(text) if len(sentence.split()) >= 6]
    duplicates = len(sentences) - len({sentence[:90].lower() for sentence in sentences})
    full_match = re.search(r"(\d+)\s+of\s+(\d+)\s+source\(s\)\s+yielded full page text", lowered)
    full_text = int(full_match.group(1)) if full_match else lowered.count("full page text extracted")
    domains = {_domain(url) for url in unique_urls}
    low_authority = [
        domain
        for domain in domains
        if any(key in domain for key in ("youtube", "facebook", "reddit", "quora", "tiktok", "pinterest"))
    ]
    words = len(text.split())
    return {
        "text": text,
        "words": words,
        "headings": headings,
        "sections_present": [name for name in required if any(name in section for section in section_names) or name in lowered],
        "findings": findings,
        "urls": urls,
        "unique_urls": unique_urls,
        "numbers": numbers,
        "vague": vague,
        "duplicates": max(duplicates, 0),
        "finding_words": sum(len(item.split()) for item in findings) / max(len(findings), 1),
        "full_text": full_text,
        "low_authority": low_authority,
    }


def build_feedback(report):
    m = _metrics(report)
    score = 10
    issues = []

    if m["words"] < 350:
        score -= 3
        issues.append(f"the report is only {m['words']} words, well below a research-report depth of 700+")
    elif m["words"] < 700:
        score -= 1
        issues.append(f"the report has {m['words']} words; another 200-300 words of analysis would help")

    if len(m["findings"]) < 5:
        score -= 2
        issues.append(f"only {len(m['findings'])} numbered findings were provided (at least 5 expected)")
    elif len(m["findings"]) < 7:
        score -= 1
        issues.append(f"{len(m['findings'])} findings are present; 7+ would give broader coverage")

    if len(m["unique_urls"]) < 3:
        score -= 2
        issues.append(f"only {len(m['unique_urls'])} unique source URL(s) are cited")
    elif len(m["unique_urls"]) < 5:
        score -= 1
        issues.append(f"{len(m['unique_urls'])} unique sources is a narrow evidence base")

    missing = [name for name in ("executive summary", "introduction", "key findings", "conclusion", "sources") if name not in m["sections_present"]]
    if missing:
        score -= min(2, len(missing))
        issues.append("missing section(s): " + ", ".join(missing))

    if "limitations and caveats" not in m["sections_present"] and "limitations" not in m["text"].lower():
        score -= 1
        issues.append("no dedicated limitations section, so uncertainty is not made explicit")

    if not m["numbers"]:
        score -= 1
        issues.append("no dates, percentages or other quantitative evidence were included")

    if m["finding_words"] and m["finding_words"] < 40:
        score -= 1
        issues.append(f"findings average only {int(m['finding_words'])} words each; explain the evidence more fully")

    if len(m["vague"]) >= 3:
        score -= 1
        issues.append("vague wording such as " + ", ".join(f"'{word}'" for word in m["vague"][:3]))

    if m["duplicates"] > 2:
        score -= 1
        issues.append(f"about {m['duplicates']} repeated sentence(s) reduce clarity")

    unique_domains = len({_domain(url) for url in m["unique_urls"]})
    if unique_domains <= 2:
        score -= 1
        issues.append("sources come from very few domains, which limits viewpoint diversity")

    if m["full_text"] == 0:
        score -= 2
        issues.append("no source page could be read in full; every claim rests on a short search snippet")
    elif m["full_text"] < 2:
        score -= 1
        issues.append(f"only {m['full_text']} source page(s) were read in full")

    if m["low_authority"]:
        score -= 1
        issues.append("low-authority domains in the source list: " + ", ".join(m["low_authority"]))

    score = max(2, min(score, 10))
    if m["full_text"] == 0:
        score = min(score, 7)
    if score == 10 and not (m["words"] >= 900 and len(m["findings"]) >= 7 and len(m["unique_urls"]) >= 8 and m["numbers"]):
        score = 9

    strengths = []
    if m["words"] >= 700:
        strengths.append(f"Substantial depth at {m['words']} words.")
    elif m["words"] >= 350:
        strengths.append(f"Reasonable length at {m['words']} words.")
    if len(m["findings"]) >= 5:
        strengths.append(f"{len(m['findings'])} distinct findings give the report structure.")
    if len(m["unique_urls"]) >= 3:
        strengths.append(f"{len(m['unique_urls'])} unique source URL(s) make claims traceable.")
    if "limitations and caveats" in m["sections_present"] or "limitations" in m["text"].lower():
        strengths.append("The report states its evidence limitations explicitly.")
    if m["numbers"]:
        strengths.append(f"{len(m['numbers'])} date/quantity reference(s) add specificity.")
    if not strengths:
        strengths.append("The report establishes a clear topic and a basic structure.")

    improvements = issues[:4] or ["Add primary-source evidence and quantify the key claims."]
    if len(issues) > 4:
        improvements.append(f"{len(issues) - 4} further issue(s) were also identified.")

    if score >= 9:
        verdict = "Exceptional, evidence-rich report with only minor room for refinement."
    elif score >= 8:
        verdict = "Strong report; a few targeted revisions would make it publication-ready."
    elif score >= 6:
        verdict = "Useful and readable, but the evidence base and depth need strengthening."
    elif score >= 4:
        verdict = "A rough draft that needs substantially more sourced, specific detail."
    else:
        verdict = "Insufficiently supported; rebuild the report around verifiable sources."

    return (
        f"Score: {score}/10\n\n"
        f"Strengths:\n" + "\n".join(f"- {item}" for item in strengths) + "\n\n"
        f"Areas to Improve:\n" + "\n".join(f"- {item}" for item in improvements) + "\n\n"
        f"One line verdict:\n{verdict}\n"
    )
