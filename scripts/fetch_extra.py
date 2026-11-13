# /// script
# requires-python = ">=3.11"
# ///
"""Extra OpenStreetMap data for the named places (scripts/landmarks.py): everything with a name and a use, and the
neighbourhoods, waterways, motorways, bridges and locks. Small, so it has its own quick run on GitHub
(.github/workflows/fetch-extra.yml), which puts the result on the data-extra branch.

Output: data/extra/osm_pois.json, data/extra/osm_named.json, data/extra/osm_villages.json
"""

from config import ROOT
from fetch import osm_bbox, overpass

EXTRA = ROOT / "data" / "extra"


def main():
    EXTRA.mkdir(parents=True, exist_ok=True)
    bbox = osm_bbox()
    keys = ("amenity", "shop", "leisure", "office", "craft", "healthcare", "man_made", "railway", "building", "club", "sport", "industrial", "landuse", "natural")
    overpass(EXTRA / "osm_pois.json", f"""[out:json][timeout:120];
( {" ".join(f'nwr["name"]["{k}"]({bbox});' for k in keys)} );
out center tags;""", "named places")
    overpass(EXTRA / "osm_named.json", f"""[out:json][timeout:120];
( nwr["place"~"^(suburb|neighbourhood|quarter|locality)$"]({bbox});
  way["waterway"]["name"]({bbox}); nwr["lock"]({bbox}); nwr["waterway"~"^(lock_gate|weir|dock)$"]({bbox});
  way["highway"~"^(motorway|trunk)$"]({bbox}); nwr["highway"="motorway_junction"]({bbox});
  way["bridge"]["name"]({bbox}); way["railway"="rail"]({bbox}); nwr["aeroway"]({bbox});
  relation["boundary"="administrative"]["admin_level"~"^(9|10|11)$"]({bbox}); );
out center tags;""", "neighbourhoods and infrastructure")
    overpass(EXTRA / "osm_villages.json", f"""[out:json][timeout:120];
( nwr["place"~"^(village|hamlet|isolated_dwelling|town|farm)$"]({bbox});
  nwr["leisure"="nature_reserve"]({bbox}); nwr["boundary"="protected_area"]({bbox}); nwr["natural"~"^(wood|heath|water|wetland)$"]["name"]({bbox});
  nwr["landuse"="forest"]["name"]({bbox}); nwr["historic"]({bbox}); nwr["heritage"]({bbox}); nwr["building"="farm"]["name"]({bbox}); );
out center tags;""", "villages and nature")

    main()
