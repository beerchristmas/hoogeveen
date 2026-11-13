# /// script
# requires-python = ">=3.11"
# ///
"""Photos for the named places, from Wikimedia Commons. Runs on GitHub (.github/workflows/fetch-photos.yml), for
build environments that cannot reach Wikimedia themselves, and puts the result on the data-photos branch.

Two jobs. For every place in photo_places.json it collects candidates: the image of its Wikidata item, the Commons
photos taken within 80 m, and a search on its name; each with licence, author and a small preview, to choose from by
eye. And it downloads the photos already chosen in photo_chosen.json ({place: Commons file name}) at full size.

Output: photos/candidates.json, photos/c/<place>-<n>.jpg, photos/chosen.json, photos/chosen/<md5>.jpg
"""

import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "photos"
UA = {"User-Agent": "isometric-hoogeveen/0.1 (https://github.com/beerchristmas/hoogeveen; personal map)"}
API = "https://commons.wikimedia.org/w/api.php"
FREE = re.compile(r"^(CC0|CC BY|CC BY-SA|Public domain|PD|No restrictions)", re.I)


def get(url, params=None, tries=5):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(tries):
        try:
            data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
            time.sleep(0.4)
            return data
        except Exception as e:  # noqa: BLE001
            print(f"{url[:120]}: {e}", file=sys.stderr)
            time.sleep(10 * (attempt + 1))
    return None


def text(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", str(html or ""))).strip()


def info(pages):
    """Title, licence, author, description and preview of Commons file pages."""
    out = []
    for p in sorted(pages, key=lambda p: p.get("index", 0)):
        ii = (p.get("imageinfo") or [{}])[0]
        meta = {k: text(v.get("value")) for k, v in (ii.get("extmetadata") or {}).items()}
        if not ii.get("thumburl") or not re.search(r"\.(jpe?g|png|tiff?)$", p["title"], re.I):
            continue
        out.append({"file": p["title"].removeprefix("File:"), "licence": meta.get("LicenseShortName", ""), "author": meta.get("Artist", "")[:120],
                    "credit": meta.get("Credit", "")[:120], "desc": meta.get("ImageDescription", "")[:300], "date": meta.get("DateTimeOriginal", "")[:40],
                    "thumb": ii["thumburl"], "w": ii.get("width"), "h": ii.get("height")})
    return out


def query(**params):
    data = get(API, {"action": "query", "format": "json", "prop": "imageinfo", "iiprop": "url|extmetadata|size", "iiurlwidth": 480, **params})
    return info((json.loads(data).get("query", {}).get("pages", {}) if data else {}).values())


def candidates(place):
    found = []
    if place.get("wikidata"):
        data = get("https://www.wikidata.org/w/api.php", {"action": "wbgetclaims", "format": "json", "entity": place["wikidata"], "property": "P18"})
        for claim in (json.loads(data).get("claims", {}).get("P18", []) if data else []):
            found += query(titles="File:" + claim["mainsnak"]["datavalue"]["value"])
    # The searches first: what a place is called says more than what happens to stand within 80 m of it.
    for words in place.get("search") or [f'"{place["name"]}" Hoogeveen']:
        found += query(generator="search", gsrsearch=words, gsrnamespace=6, gsrlimit=5)
    lat, lon = place["latlon"]
    found += query(generator="geosearch", ggscoord=f"{lat}|{lon}", ggsradius=80, ggsnamespace=6, ggslimit=6)
    seen, out = set(), []
    for c in found:
        if c["file"] not in seen and FREE.match(c["licence"]):
            seen.add(c["file"])
            out.append(c)
    return out[:8]


def main():
    (OUT / "c").mkdir(parents=True, exist_ok=True)
    (OUT / "chosen").mkdir(parents=True, exist_ok=True)
    result = {}
    queries = json.loads((HERE / "photo_queries.json").read_text()) if (HERE / "photo_queries.json").exists() else {}
    chosen_path = HERE / "photo_chosen.json"
    chosen = json.loads(chosen_path.read_text()) if chosen_path.exists() else {}
    todo = [{**p, "search": queries.get(p["name"])} for p in json.loads((HERE / "photo_places.json").read_text()) if p["name"] not in chosen]
    for n, place in enumerate(todo):
        try:
            cands = candidates(place)
        except Exception as e:  # noqa: BLE001
            print(f"{place['name']}: {e}", file=sys.stderr)
            cands = []
        for k, c in enumerate(cands):
            data = get(c.pop("thumb"))
            if data:
                c["preview"] = f"c/{n:03d}-{k}.jpg"
                (OUT / c["preview"]).write_bytes(data)
        result[place["name"]] = cands
        print(f"{place['name']}: {len(cands)}", file=sys.stderr)
    (OUT / "candidates.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))

    meta = {}
    for name, file in chosen.items():
        got = query(titles="File:" + file, iiurlwidth=960)
        if not got:
            continue
        data = get(got[0].pop("thumb"))
        if data:
            (OUT / "chosen" / f"{hashlib.md5(file.encode()).hexdigest()[:12]}.jpg").write_bytes(data)
            meta[name] = got[0]
    (OUT / "chosen.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
