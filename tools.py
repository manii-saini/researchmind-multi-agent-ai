import os
import re
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv(override=True)

TAVILY_KEY = os.getenv("TAVILY_API_KEY", "").strip()
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchMind/1.0)"}

OFFLINE_NOTES = {
    "quantum": [
        {
            "title": "Quantum computing - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Quantum_computing",
            "content": "Quantum computing uses qubits, which exploit superposition and entanglement to process information. Error correction and scaling remain the main engineering challenges.",
        },
        {
            "title": "Quantum Computing - IBM",
            "url": "https://www.ibm.com/quantum",
            "content": "IBM offers cloud-accessible quantum processors and a software stack for building and running quantum circuits on real hardware and simulators.",
        },
        {
            "title": "Quantum computing - Nature",
            "url": "https://www.nature.com/subjects/quantum-computing",
            "content": "Peer-reviewed research covers quantum algorithms, hardware platforms, error correction and applications in chemistry and materials science.",
        },
    ],
    "crispr": [
        {
            "title": "CRISPR - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/CRISPR",
            "content": "CRISPR is a genome-editing technology derived from bacterial immune systems. It allows targeted changes to DNA using a guide RNA and a nuclease such as Cas9.",
        },
        {
            "title": "CRISPR gene editing - Nature",
            "url": "https://www.nature.com/subjects/crispr-cas9",
            "content": "Research covers precision, delivery methods, off-target effects and therapeutic applications in genetics and medicine.",
        },
        {
            "title": "Broad Institute - CRISPR",
            "url": "https://www.broadinstitute.org/crispr",
            "content": "The Broad Institute publishes open resources and research on CRISPR-based screens, diagnostics and genome engineering.",
        },
    ],
    "fusion": [
        {
            "title": "Nuclear fusion - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Nuclear_fusion",
            "content": "Nuclear fusion joins light atomic nuclei to release energy. Magnetic confinement and inertial confinement are the two main research approaches.",
        },
        {
            "title": "ITER - International Thermonuclear Experimental Reactor",
            "url": "https://www.iter.org",
            "content": "ITER is a large international tokamak project aimed at demonstrating net energy gain from magnetically confined plasma.",
        },
        {
            "title": "Fusion energy - LLNL",
            "url": "https://lasers.llnl.gov/science/fusion",
            "content": "Lawrence Livermore National Laboratory studies inertial confinement fusion using high-power lasers at the National Ignition Facility.",
        },
    ],
    "llm": [
        {
            "title": "Large language model - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Large_language_model",
            "content": "Large language models are neural networks trained on large text corpora to predict and generate language. They are used in chat, summarization and code generation.",
        },
        {
            "title": "Language agents - arXiv",
            "url": "https://arxiv.org/list/cs.AI/recent",
            "content": "Recent research explores agents that combine language models with tools, memory and planning loops to complete multi-step tasks.",
        },
        {
            "title": "LangChain documentation",
            "url": "https://python.langchain.com/docs/introduction/",
            "content": "LangChain provides building blocks for chaining prompts, models, tools and retrieval steps into applications.",
        },
    ],
    "agent": [
        {
            "title": "AI agent - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Intelligent_agent",
            "content": "An intelligent agent perceives its environment and takes actions to maximize its chances of achieving a goal.",
        },
        {
            "title": "LangGraph documentation",
            "url": "https://langchain-ai.github.io/langgraph/",
            "content": "LangGraph models agent workflows as graphs, enabling durable, stateful multi-step applications.",
        },
    ],
}


def _tavily_search(query, max_results):
    if not TAVILY_KEY:
        return []
    try:
        from tavily import TavilyClient

        raw = TavilyClient(api_key=TAVILY_KEY).search(
            query=query, max_results=max_results
        )
        return [
            {
                "title": item.get("title") or item.get("url", ""),
                "url": item.get("url", ""),
                "content": item.get("content") or "",
            }
            for item in raw.get("results", [])
            if item.get("url")
        ][:max_results]
    except Exception:
        return []


def _duckduckgo_search(query, max_results):
    try:
        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers=HEADERS,
            timeout=10,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for item in soup.select(".result")[:max_results]:
            link = item.select_one("a.result__a")
            if not link or not link.get("href"):
                continue
            url = link["href"]
            if "uddg=" in url:
                url = unquote(
                    parse_qs(urlparse(url).query).get("uddg", [url])[0]
                )
            snippet = item.select_one(".result__snippet")
            results.append(
                {
                    "title": link.get_text(" ", strip=True),
                    "url": url,
                    "content": snippet.get_text(" ", strip=True) if snippet else "",
                }
            )
        return results
    except Exception:
        return []


def _wikipedia_search(query, max_results):
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
            },
            headers=HEADERS,
            timeout=10,
        )
        response.raise_for_status()
        hits = response.json().get("query", {}).get("search", [])
        return [
            {
                "title": hit.get("title", ""),
                "url": "https://en.wikipedia.org/wiki/"
                + quote(hit.get("title", "").replace(" ", "_")),
                "content": BeautifulSoup(
                    hit.get("snippet", ""), "html.parser"
                ).get_text(" ", strip=True),
            }
            for hit in hits
        ]
    except Exception:
        return []


def _offline_search(query, max_results):
    lowered = query.lower()
    matches = []
    for keyword, notes in OFFLINE_NOTES.items():
        if keyword in lowered:
            matches.extend(notes)
    seen = set()
    unique = []
    for item in matches:
        if item["url"] not in seen:
            seen.add(item["url"])
            unique.append(item)
    if unique:
        return unique[:max_results]
    search_url = (
        "https://en.wikipedia.org/w/index.php?search=" + quote(query)
    )
    return [
        {
            "title": f"Reference overview: {query}",
            "url": search_url,
            "content": (
                "Offline reference mode is active. The topic was matched against a "
                "built-in knowledge index; connect to a network for live web results "
                "or add a Tavily key to enrich the research."
            ),
        }
    ]


def _search(query, max_results=5):
    for provider in (
        _tavily_search,
        _duckduckgo_search,
        _wikipedia_search,
    ):
        results = provider(query, max_results)
        if results:
            return results
    return _offline_search(query, max_results)


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic. Returns titles, URLs and snippets."""
    results = _search(query, 7)
    blocks = []
    for item in results:
        content = re.sub(r"\s+", " ", item.get("content") or "").strip()
        blocks.append(
            f"Title: {item.get('title', '')}\n"
            f"URL: {item.get('url', '')}\n"
            f"Snippet: {content[:450]}\n"
        )
    return "\n----\n".join(blocks)


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        response = requests.get(
            url, timeout=10, headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000] or (
            "The page returned no readable text."
        )
    except Exception as exc:
        return f"Could not scrape URL: {type(exc).__name__}: {exc}"
