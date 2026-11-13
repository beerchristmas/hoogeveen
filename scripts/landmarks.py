# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Named places of Hoogeveen: the viewer's place list and the stories on its info cards.

Two sources. CURATED holds the places with a story, found by their BAG address or a position; the facts come from
the Rijksmonumentenregister (RCE), archimon.nl, standbeelden.vanderkrogt.net, visitdrenthe.com (city walk
Trommelslager), binnenstadhoogeveen.nl, hoogeveen2000.com (Albert Metselaar), vangoghhuis.nl, cascaderun.nl,
artatsite.com and landschaplopen.com. The rest comes from OpenStreetMap (data/raw/osm_landmarks.json): churches,
artworks, memorials, museums, graveyards and the like, with their name and kind and no story.

An entry may also carry style overrides for the render ("style", "wall", "roof", "tower", ...); none do yet.

Output: data/build/landmarks.json
"""

import json

import numpy as np

from config import BUILD, RAW, XMAX, XMIN, YMAX, YMIN
from geo import wgs_to_rd

RM = "Rijksmonument"

# Places with a story. "addr" is a BAG address (the building under it becomes the hover area), "latlon" a point,
# "osm" the name of the OpenStreetMap element whose position to use (and which is then not listed twice).
CURATED = [
    {
        "name": "Grote Kerk",
        "addr": "Grote Kerkstraat 40",
        "kind": f"{RM} · 17e eeuw",
        "note": "Kruiskerk uit de 17e eeuw, later uitgebouwd tot hallenkerk",
        "address": "Grote Kerkstraat 40",
        "osm": ["Grote Kerk"],
        "text": "De bouw begon rond 1652, als kruiskerk met vier even lange armen. Tussen 1766 en 1804 is hij in drie stappen uitgebouwd tot een driebeukige hallenkerk. Hoogeveen werd niet met klokken maar met een trom naar de kerk geroepen. Het bekende verhaal dat de veengrond geen toren kon dragen klopt volgens lokaal historicus Albert Metselaar niet: de eerste kerk had een toren, maar klokken vond men te duur, en toen er later klokken werden aangeboden wees men ze af, want er was al een trommelslager. In de klokkenstoel hangt een klok van F. en P. Hemony uit 1649. Boven de zuidingang zit een houten zonnewijzer, gesigneerd Hend. Casp. Dekker, 1804.",
    },
    {
        "name": "Trommelslager",
        "osm": ["Trommelslager"],
        "kind": "Standbeeld · 1971",
        "note": "Beeld van Johan Sterenberg",
        "text": "Beeld van Johan Sterenberg uit 1971. De trommelslager riep Hoogeveen elke zondag naar de kerk en rukte ook uit bij brand. Waar het gebruik vandaan komt is niet zeker. De waarschijnlijkste verklaring is een militaire: Roelof van Echten, de man achter de vervening, was kapitein, en in Heerenveen begon en eindigde het werk in het veen in 1622 ook al op de trom. In 1754 is er over gepraat om de trommelslager af te schaffen. Het is er niet van gekomen. Het beeld is een keer gestolen en staat weer op zijn sokkel.",
    },
    {
        "name": "Remonstrantse kerk",
        "addr": "Grote Kerkstraat 39",
        "kind": "Kerk · 1894",
        "address": "Grote Kerkstraat 39",
        "osm": ["Remonstrantse Kerk"],
        "text": "Gebouwd in 1893 en 1894 in neorenaissancestijl, naar ontwerp van gemeentearchitect J. Carmiggelt.",
    },
    {
        "name": "Hoofdstraatkerk",
        "addr": "Hoofdstraat 95",
        "kind": "Kerk · 1904",
        "address": "Hoofdstraat 95",
        "osm": ["Protestantse Gemeente Hoogeveen"],
        "text": "Gereformeerde kerk uit 1904 van architect G. Hoekzema, in een eigenzinnige neoromaanse stijl met hoektorens. Hoog in de gevel staat een steen met de jaartallen 1834-1886: de Afscheiding en de Doleantie, de twee keren dat groepen de Hervormde Kerk verlieten.",
    },
    {
        "name": "Gemeentehuis",
        "addr": "Raadhuisplein 1",
        "kind": f"{RM} · 1940",
        "note": "Raadhuis met carillontoren",
        "address": "Raadhuisplein 1",
        "text": "Raadhuis uit 1940 van architect C.J. Blaauw, in de stijl van de Delftse School, met een toren waarin een carillon hangt. Het is gebouwd met het geld dat overbleef toen de Compagnie van de 5000 Morgen, de veenmaatschappij waar Hoogeveen mee begon, werd opgeheven. De laatste NSB-burgemeester, Jan Marinus Veldhuis, was na een spoedcursus benoemd en zat er van juli 1944 tot april 1945. Toen de Canadezen kwamen vluchtte hij op de fiets en verstopte hij zich in een wc. Op het Raadhuisplein start elk voorjaar de Cascaderun.",
    },
    {
        "name": "Schippershuus",
        "addr": "Hoofdstraat 254",
        "kind": f"{RM} · 17e eeuw",
        "note": "Rentmeesterskantoor van de Compagnie, later schipperscafé",
        "address": "Hoofdstraat 254",
        "text": "Het oudste gebouw van Hoogeveen, met een kern uit 1632 of 1633. De Compagnie van de 5000 Morgen liet het bouwen als kantoor van de rentmeester, waar de bewoners hun pacht kwamen betalen. De eerste rentmeester die we kennen is Carst Peters uit Hasselt. Na een verbouwing in 1882 werd het een café waar de turfschippers samenkwamen. Op het pleintje erachter staat een stenen huisje van één kamer dat doorgaat voor het oudste huis van Hoogeveen, uit 1630. Twee vrijwilligers van het museum hebben het in 2006 bekeken: het metselwerk is twintigste-eeuws en in het register van 1758 komt het huisje niet voor.",
    },
    {
        "name": "Huis met de Duivengaten",
        "addr": "Alteveerstraat 1",
        "kind": f"{RM} · eind 17e eeuw",
        "address": "Alteveerstraat 1",
        "text": "Huis uit het eind van de 17e eeuw, met een gevel vol duivengaten, waterlijsten en pinakels. Het staat aan Het Kruis, de plek waar Hoogeveen begon. Nu is het een restaurant.",
    },
    {
        "name": "Het Kruis",
        "latlon": (52.72056, 6.47778),
        "kind": "Hier begon Hoogeveen",
        "text": "Op deze kruising van de Hoofdstraat, Het Haagje, de Alteveerstraat en de Schutstraat kwamen twee vaarten samen. Hier vestigden zich de eerste winkeliers, turfhandelaren en ambachtslieden. Op de kaart van 1637 liep de vaart die nu Het Haagje is door tot aan het Groote Meer, dat verdwenen is. De vaarten zijn gedempt.",
    },
    {
        "name": "Turfschipper",
        "osm": ["Turfschipper"],
        "kind": "Standbeeld · 1982",
        "note": "Schippersmonument van Arie Teeuwisse",
        "text": "Bronzen turfschipper van Arie Teeuwisse, met op zijn rug het anker waarmee hij zijn schip vastlegde. Commissaris der Koningin A.P. Oele onthulde het beeld op 24 november 1982. Het stond eerst bij het Kerkplein en is verhuisd toen de Cascade werd aangelegd.",
    },
    {
        "name": "De Cascade",
        "latlon": (52.723544, 6.477679),
        "kind": "Waterkunstwerk",
        "note": "Waterkunstwerk van Karin Daan in de Hoofdstraat",
        "text": "Waterkunstwerk van Karin Daan, zo'n 250 meter lang en 10 meter breed, midden in de Hoofdstraat. Hier liep de vaart waarover de turf werd afgevoerd; het laatste stuk is in 1972 gedempt. De Cascade wordt het langste kunstwerk van Europa genoemd. Volgens Albert Metselaar zei de ontwerper zelf dat ze zich had laten inspireren door oosterse tuinen, niet door het veenverleden. Hij noemt het met een knipoog het grootste vruchtbaarheidssymbool van Nederland. De Cascaderun, waarvan de finish hier ligt, is ernaar genoemd.",
    },
    {
        "name": "De Tas",
        "osm": ["De Tas"],
        "kind": "Kunstwerk · 1999",
        "text": "Boodschappentas van Gert-Jan Mulder bij de ingang van de Tamboerpassage, geplaatst toen het winkelcentrum in 1999 openging. Hij kostte 95.000 gulden.",
    },
    {
        "name": "Schouwburg De Tamboer",
        "addr": "Hoofdstraat 17",
        "kind": "Theater",
        "address": "Hoofdstraat 17",
        "web": "https://www.detamboer.nl",
        "text": "De schouwburg is genoemd naar de trommelslager. Bij het gebouw staan De Tamboer van Gooitzen de Jong uit 1967 en vier figuren uit de commedia dell'arte van Arie Teeuwisse uit 1989: Colombina, Pantalone, Pulcinella en Scapino.",
    },
    {
        "name": "Logement Hartsuiker",
        "addr": "Pesserstraat 24",
        "kind": "Van Gogh sliep hier · 1883",
        "note": "Hier logeerde Vincent van Gogh in 1883",
        "address": "Pesserstraat 24",
        "web": "https://logementhartsuiker.nl",
        "text": "Vincent van Gogh kwam op dinsdag 11 september 1883 om negen uur 's avonds met de trein in Hoogeveen aan. Spoorwegarbeider Albertus Hartsuiker bood hem kost en inwoning aan voor een gulden per dag. Hij bleef een paar weken en trok daarna verder Drenthe in. Lang was niet duidelijk welk huis het was; sinds onderzoek van Albert Metselaar uit 2001 geldt dit logement aan de Pesserstraat, toen de Toldijk, als het adres. Op de gevel zit een plaquette.",
    },
    {
        "name": "Station Hoogeveen",
        "addr": "Stationsplein 1",
        "kind": "Station",
        "address": "Stationsplein 1",
        "text": "De spoorlijn van Meppel naar Groningen ging op 1 mei 1870 open, met een station in Hoogeveen. Het gebouw van nu is uit 1984 en ontworpen door Cees Douma.",
    },
    {
        "name": "Het Klooster",
        "addr": "Brinkstraat 5",
        "kind": "Voormalig klooster · ca. 1905",
        "address": "Brinkstraat 5",
        "osm": ["Het Klooster"],
        "text": "De paters karmelieten uit Zenderen lieten dit neogotische gebouw ontwerpen door architect Croonen uit Almelo. Het kwam rond 1905 in gebruik. Vanuit Hoogeveen bedienden ze de katholieken die verspreid over de streek woonden. In 1957 vertrokken ze.",
    },
    {
        "name": "Dubbel woonhuis Brinkstraat",
        "addr": "Brinkstraat 32",
        "kind": f"{RM} · 1904",
        "address": "Brinkstraat 32 en 34",
        "text": "Dubbel woonhuis uit 1904, ontworpen door G. Hofman voor mevrouw J. van Hees, met art-nouveaudetails: glas-in-lood, bloemmotieven en mozaïeken van baksteen. Het stond aan de Hoogeveensche Vaart, die hier nu een straat is.",
    },
    {
        "name": "Voormalige synagoge",
        "addr": "Schutstraat 147",
        "kind": "Voormalige synagoge",
        "address": "Schutstraat 147",
        "osm": ["Baptistengemeente de Schutse"],
        "text": "De synagoge van Hoogeveen. Nu kerkt de baptistengemeente De Schutse er.",
    },
    {
        "name": "Molen De Zwaluw",
        "addr": "Van Echtenstraat 47",
        "kind": "Korenmolen · 1834",
        "address": "Van Echtenstraat 47",
        "text": "Korenmolen uit 1834, de opvolger van een molen die in 1787 afbrandde. De naam kreeg hij pas na de restauratie van 1980. Hij maalt nog: als het niet waait op een elektromotor. Het meel is te koop in de molenwinkel.",
    },
    {
        "name": "Oude Joodse Begraafplaats",
        "osm": ["Oude Joodse Begraafplaats"],
        "kind": "Begraafplaats · 18e eeuw",
        "text": "Joodse begraafplaats uit de 18e eeuw, achter de oude pastorie aan de Van Echtenstraat. De bronnen noemen 1725 en 1730 als begin; hij was in gebruik tot 1839. Daarna kwam de begraafplaats aan de Zuiderweg, uit 1829.",
    },
    {
        "name": "De Hoogeveense Kei",
        "osm": ["De Hoogeveense Kei"],
        "kind": "Zwerfkei",
        "text": "Zwerfkei die in de ijstijd uit Scandinavië is meegekomen en in het Hollandscheveld is gevonden. Toen de kei in 2008 werd verplaatst, kwam er een loden koker onder vandaan met een oorkonde van 14 juni 1955, gericht aan het nageslacht in het atoomtijdperk. Dat schrijft de wandelaar van landschaplopen.com, die het verhaal bij de kei zelf vond.",
    },
    {
        "name": "De kerk op de rotonde",
        "osm": ["De kerk op de rotonde"],
        "kind": "Kunstwerk · 2008",
        "text": "Kunstwerk van Filip Jonker uit 2008, op de rotonde van de Middenveldweg en de A.G. Bellstraat.",
    },
    {
        "name": "De Turfgravers",
        "osm": ["Turfgravers"],
        "kind": "Beeld · 1975",
        "text": "Beeld van Han Wezelaar uit 1975, in het Burgemeester Tjalmapark.",
    },
    {
        "name": "Piet van de Velde",
        "osm": ["Pieter Cornelis Marinus van de Velde (1897-1945)"],
        "kind": "Standbeeld · 2010",
        "text": "Beeld van Daphné Du Barry uit 2010, bij het ziekenhuis.",
    },
    {
        "name": "Ziekenhuis Bethesda",
        "addr": "Dr G H Amshoffweg 1",
        "kind": "Ziekenhuis",
        "address": "Dr. G.H. Amshoffweg 1",
        "text": "Rond het ziekenhuis staan vijf beelden: De Haan van Frank Letterie (1970), De Belhamel van Adri de Waard (1988), De Bron van Meri Matacin (1989), De Turenden van Sjer Jacobs (2002) en Piet van de Velde van Daphné Du Barry (2010).",
    },
    {
        "name": "De Beurs",
        "addr": "Grote Kerkstraat 28",
        "kind": "Voormalig postkantoor",
        "address": "Grote Kerkstraat 28",
        "text": "Eetcafé op de plek van het oude postkantoor. Op 17 februari 1944 overviel het verzet het postkantoor van Hoogeveen en nam zo'n 13.000 bonkaarten mee, voor onderduikers. Loketbeambte Henk Raak is later opgepakt en in de duinen bij Overveen gefusilleerd.",
    },
    {
        "name": "Bibliotheek Hoogeveen",
        "addr": "Willemskade 27",
        "kind": "Bibliotheek",
        "address": "Willemskade 27",
        "text": "Museum De 5000 Morgen sloot in 2015 en ging op in de bibliotheek. Een deel van de collectie is hier te zien, met een replica van de kano van Pesse, de oudste boot ter wereld, die een paar kilometer verderop is gevonden.",
    },
    {
        "name": "Velden van Nevel",
        "osm": ["Velden van Nevel", "Velden van nevel", "De Lichtnaald"],
        "kind": "Kunstwerk · 2007",
        "text": "Een reeks beelden van zo'n tien meter hoog langs de A28 en de A37, van Rudi van de Wint (1942-2006). Zijn zonen hebben ze na zijn dood geplaatst. De titel verwijst naar het volksgeloof in nevelheksen, de witte wieven die rond Hoogeveen over de velden zouden spoken.",
    },
]

# OpenStreetMap kinds: (tag, value) to the label on the info card and whether it is a building.
OSM_KINDS = [
    (("building", "mosque"), "Moskee", True),
    (("amenity", "place_of_worship"), "Kerk", True),
    (("tourism", "museum"), "Museum", True),
    (("tourism", "hotel"), "Hotel", True),
    (("memorial", "war_memorial"), "Oorlogsmonument", False),
    (("memorial", "plaque"), "Plaquette", False),
    (("historic", "memorial"), "Monument", False),
    (("artwork_type", "statue"), "Standbeeld", False),
    (("artwork_type", "mural"), "Muurschildering", False),
    (("tourism", "artwork"), "Kunstwerk", False),
    (("tourism", "zoo"), "Dierenweide", False),
    (("tourism", "viewpoint"), "Uitkijkpunt", False),
    (("tourism", "caravan_site"), "Camperplaats", False),
    (("landuse", "cemetery"), "Begraafplaats", False),
    (("leisure", "pitch"), "Sportveld", False),
]


def osm_places():
    """Named OpenStreetMap elements inside the map: {name: [(x, y, kind, is_building, tags), ...]}."""
    path = RAW / "osm_landmarks.json"
    out = {}
    if not path.exists():
        return out
    for e in json.loads(path.read_text())["elements"]:
        t = e.get("tags", {})
        if "name" not in t:
            continue
        pts = [(e["lat"], e["lon"])] if "lat" in e else [(g["lat"], g["lon"]) for g in e.get("geometry", [])]
        pts += [(g["lat"], g["lon"]) for m in e.get("members", []) for g in m.get("geometry", [])]
        if not pts:
            continue
        x, y = np.mean([wgs_to_rd(lat, lon) for lat, lon in pts], axis=0)
        if not (XMIN < x < XMAX and YMIN < y < YMAX):
            continue
        for (k, v), label, is_building in OSM_KINDS:
            if t.get(k) == v:
                year = t.get("start_date", "")[:4]
                kind = f"{label} · {year}" if year.isdigit() else label
                out.setdefault(t["name"], []).append((float(x), float(y), kind, is_building and "lat" not in e, t))
                break
    return out


def places():
    """BUILDINGS and POINTS: the curated places, then what is left of OpenStreetMap."""
    osm = osm_places()
    buildings, points, used = [], [], set()
    for c in CURATED:
        c = dict(c)
        names = c.pop("osm", [])
        used.update(names)
        if "addr" in c:
            c["at"] = address_xy(c.pop("addr"))
            buildings.append(c)
            continue
        if "latlon" not in c:
            found = [p for n in names for p in osm.get(n, [])]
            assert found, f"{c['name']}: not in OpenStreetMap"
            c["at"] = found[0][:2]
            for i, p in enumerate(found[1:], 2):  # the same work in several places
                points.append({**c, "name": f"{c['name']} ({i})", "at": p[:2]})
        points.append(c)
    for name, found in sorted(osm.items()):
        if name in used:
            continue
        kept = []
        for x, y, kind, is_building, t in found:
            if any(np.hypot(x - kx, y - ky) < 150 for kx, ky in kept):
                continue  # one sports field of several, or a way and a node for the same thing
            kept.append((x, y))
            entry = {"name": name if len(kept) == 1 else f"{name} ({len(kept)})", "at": (x, y), "kind": kind}
            if t.get("website"):
                entry["web"] = t["website"]
            (buildings if is_building else points).append(entry)
    return buildings, points


# Series in the viewer's panel: a walk past these places, starting with the one nearest the village.
CATEGORIES = []

# What the viewer shows when a place is clicked; the rest of an entry is for the render.
INFO = ("note", "kind", "text", "address", "web", "more", "photo", "stop")

# Lattice masts: {"latlon": (lat, lon), "height": m, "base": m, "tip": m}.
MASTS = []


def point_in_ring(x, y, ring):
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, -1, axis=0)):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


ADDRESSES = None


def address_xy(name):
    """RD position of a BAG address such as "Hoofdstraat 3" (data/raw/addresses.json, from scripts/fetch.py)."""
    global ADDRESSES
    if ADDRESSES is None:
        ADDRESSES = {label.replace(", Hoogeveen", "").split(", ")[0].lower(): (x, y) for x, y, label in json.loads((RAW / "addresses.json").read_text())}
    return ADDRESSES[name.lower()]


def building_at(rings, x, y):
    for rid, ring, a in rings:
        if point_in_ring(x, y, ring):
            return rid, a
    return None, None


def main():
    feats = [json.loads(line) for line in open(RAW / "3dbag.jsonl")]
    rings = []
    for f in feats:
        t = f["transform"]
        V = np.array(f["vertices"], dtype=np.float64) * t["scale"] + t["translate"]
        co = f["CityObjects"][f["id"]]
        rings.append((f["id"], V[co["geometry"][0]["boundaries"][0][0]][:, :2], co["attributes"]))

    out = {"buildings": {}, "places": [], "masts": [], "hotspots": [], "categories": []}
    BUILDINGS, POINTS = places()
    for b in BUILDINGS:
        if "addrs" in b:
            pts = [address_xy(a) for a in b["addrs"]]
            pids = [building_at(rings, *pt)[0] for pt in pts]
            assert all(pids), f"{b['name']}: no building at {b['addrs']}"
            x, y = np.mean(pts, axis=0)
            out["places"].append({"name": b["name"], "note": b.get("note"), "x": x, "y": y, "h": 4.0})
            out["hotspots"].append({"name": b["name"], "pids": pids, "x": x, "y": y, **{k: b[k] for k in INFO if k in b}})
            print(f"{b['name']}: {', '.join(pids)}")
            continue
        x, y = b["at"] if "at" in b else wgs_to_rd(*b["latlon"])
        pid, attrs = None, None
        for rid, ring, a in rings:
            if point_in_ring(x, y, ring):
                pid, attrs = rid, a
                break
        if pid is None:
            d = [(np.hypot(*(ring.mean(0) - (x, y))), rid, a) for rid, ring, a in rings]
            dist, pid, attrs = min(d, key=lambda t: t[0])
            if dist > 20:
                pid = None
        height = ((attrs or {}).get("b3_h_dak_max") or 0) - ((attrs or {}).get("b3_h_maaiveld") or 0)
        if "tower" in b:
            height = b["tower"]["top"]
        elif "turret" in b:
            height = b["turret"]["top"]
        out["places"].append({"name": b["name"], "note": b.get("note"), "x": x, "y": y, "h": round(height, 1)})
        if pid and "style" in b:
            out["buildings"][pid] = {k: v for k, v in b.items() if k not in ("at", "latlon", *INFO)}
        if pid:
            out["hotspots"].append({"name": b["name"], "pid": pid, "x": x, "y": y, **{k: b[k] for k in INFO if k in b}})
        print(f"{b['name']}: {pid or 'no building'}")
    for p in POINTS:
        x, y = p["at"] if "at" in p else address_xy(p["addr"]) if "addr" in p else wgs_to_rd(*p["latlon"])
        out["places"].append({"name": p["name"], "note": p.get("note"), "x": x, "y": y, "h": 2.0})
        out["hotspots"].append({"name": p["name"], "x": x, "y": y, **{k: p[k] for k in INFO if k in p}})
    for m in MASTS:
        x, y = wgs_to_rd(*m["latlon"])
        out["masts"].append({**{k: v for k, v in m.items() if k != "latlon"}, "x": x, "y": y})
    names = {h["name"] for h in out["hotspots"]}
    for key, label, members in CATEGORIES:
        missing = [n for n in members if n not in names]
        assert not missing, f"{label}: no hotspot for {missing}"
        out["categories"].append({"key": key, "label": label, "names": members})
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "landmarks.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
