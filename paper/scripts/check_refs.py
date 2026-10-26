"""Check every entry of refs.bib against the record of its DOI (Crossref, https://api.crossref.org/works/DOI) or its arXiv identifier (https://export.arxiv.org/api/query?id_list=ID): the title, the family name of the first author, the
year, and for a DOI the volume and the first page. Prints one line for each entry, OK or what differs, and writes refs_check.json beside the bib file. It needs the network. usage: python paper/scripts/check_refs.py
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
HEADERS = {"User-Agent": "transitgpu-reference-check (mailto:mauri870@gmail.com)"}


def get(url):
    for wait in (0, 5, 20):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as response:
                return response.read()
        except Exception as error:
            last = error
    raise last


def entries(text):
    out = []
    for block in re.split(r"\n(?=@)", text):
        m = re.match(r"@(\w+)\{([^,]+),", block.strip())
        if not m:
            continue
        fields = {}
        for name, value in re.findall(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|[\w./-]+)", block):
            fields[name.lower()] = value.strip("{}\" ")
        out.append({"key": m.group(2), "type": m.group(1), **fields})
    return out


def normal(text):
    text = re.sub(r"\\[a-zA-Z]+|[{}$\\^_~]", " ", text or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def first_family(authors):
    first = re.split(r"\s+and\s+", authors or "")[0]
    return normal(first.split(",")[0]) if "," in first else normal(first.split()[-1] if first.split() else "")


def crossref(doi):
    message = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    year = (message.get("issued") or message.get("published") or {}).get("date-parts", [[None]])[0][0]
    return {"title": (message.get("title") or [""])[0], "first": normal((message.get("author") or [{}])[0].get("family", "")), "year": year, "volume": message.get("volume"), "page": (message.get("page") or message.get("article-number") or "").split("-")[0]}


def arxiv(identifier):
    root = ElementTree.fromstring(get("https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(identifier)))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    entry = root.find("a:entry", ns)
    if entry is None or entry.find("a:title", ns) is None:
        return None
    return {"title": " ".join(entry.find("a:title", ns).text.split()), "first": normal((entry.find("a:author/a:name", ns).text or "").split()[-1]), "year": int(entry.find("a:published", ns).text[:4])}


def main():
    report = []
    for entry in entries((HERE / "refs.bib").read_text()):
        found = None
        try:
            if entry.get("doi"):
                found, source = crossref(entry["doi"]), "doi " + entry["doi"]
            elif entry.get("eprint"):
                found, source = arxiv(entry["eprint"]), "arXiv " + entry["eprint"]
            else:
                source = "no doi and no eprint"
        except Exception as error:
            source = f"lookup failed: {type(error).__name__} {str(error)[:60]}"
        problems = []
        if found is None:
            problems.append(source)
        else:
            if normal(found["title"])[:40] != normal(entry.get("title"))[:40] and not set(normal(entry.get("title")).split()) & set(normal(found["title"]).split()) >= set(normal(entry.get("title")).split()[:3]):
                problems.append(f"title: bib '{entry.get('title')}' / record '{found['title']}'")
            if found["first"] and first_family(entry.get("author")) != found["first"]:
                problems.append(f"first author: bib '{first_family(entry.get('author'))}' / record '{found['first']}'")
            if found["year"] and entry.get("year") and abs(int(entry["year"]) - int(found["year"])) > 1:
                problems.append(f"year: bib {entry['year']} / record {found['year']}")
            for key in ("volume", "page"):
                if found.get(key) and entry.get(key if key == "volume" else "pages") and str(found[key]) not in entry.get(key if key == "volume" else "pages"):
                    problems.append(f"{key}: bib {entry.get(key if key == 'volume' else 'pages')} / record {found[key]}")
        report.append({"key": entry["key"], "source": source, "problems": problems, "record": found})
        print(entry["key"], "OK" if not problems else "; ".join(problems), f"({source})", flush=True)
        time.sleep(1)
    (HERE / "refs_check.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0 if all(not r["problems"] for r in report) else 1


if __name__ == "__main__":
    sys.exit(main())
