import re

from llm import FEEDBACK_PROMPT, REPORT_PROMPT, generate
from tools import scrape_url, web_search


class _Message:
    def __init__(self, content):
        self.content = content
        self.type = "ai"


class _Chain:
    def __init__(self, runner):
        self._runner = runner

    def invoke(self, payload):
        return self._runner(payload)


def build_llm():
    return generate


def backend_info():
    from llm import backend_info as _backend_info

    return _backend_info()


def _last_user_text(payload):
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, tuple):
        return str(last[-1])
    if isinstance(last, dict):
        return str(last.get("content", ""))
    return str(getattr(last, "content", ""))


def _extract_query(text):
    match = re.search(r"about:\s*(.+)$", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _extract_urls(text):
    urls = re.findall(r"URL:\s*(https?://[^\s]+)", text, re.IGNORECASE)
    if not urls:
        urls = re.findall(r"https?://[^\s)]+", text)
    return [url.rstrip("),.;'\"") for url in urls[:5]]


class _ToolAgent:
    def __init__(self, tool_fn, kind):
        self._tool_fn = tool_fn
        self._kind = kind

    def invoke(self, payload):
        text = _last_user_text(payload)
        if self._kind == "search":
            query = _extract_query(text)
            content = self._tool_fn.invoke({"query": query})
        else:
            urls = _extract_urls(text)
            if not urls:
                content = (
                    "No URL was found in the search results. "
                    "Run the search step again with a more specific topic."
                )
            else:
                blocks = []
                for url in urls:
                    scraped = self._tool_fn.invoke({"url": url})
                    lowered = scraped.lower()
                    if scraped.startswith("Could not scrape") or any(
                        marker in lowered
                        for marker in (
                            "checking your browser",
                            "recaptcha",
                            "enable javascript",
                            "access denied",
                            "just a moment",
                        )
                    ):
                        continue
                    blocks.append(f"Source: {url}\n\n{scraped}")
                content = "\n\n----\n\n".join(blocks) or (
                    "The top sources blocked automated reading, but their search "
                    "snippets are still available for the report."
                )
        return {"messages": [_Message(content)]}


def build_search_agent():
    return _ToolAgent(web_search, "search")


def build_reader_agent():
    return _ToolAgent(scrape_url, "reader")


def build_writer_chain():
    def run(payload):
        topic = payload.get("topic", "")
        research = payload.get("research", "")
        prompt = REPORT_PROMPT.format(topic=topic, research=research)
        return generate(prompt, task="report", topic=topic, research=research)

    return _Chain(run)


def build_critic_chain():
    def run(payload):
        report = payload.get("report", "")
        prompt = FEEDBACK_PROMPT.format(report=report)
        return generate(prompt, task="feedback", report=report)

    return _Chain(run)
