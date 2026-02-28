from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TypedDict

NCOLS = 26
NROWS = 34

LANDUSE_FLAGS = [
    "residential",
    "grass",
    "retail",
    "construction",
    "industrial",
    "recreation_ground",
    "commercial",
    "railway",
    "allotments",
    "religious",
    "flowerbed",
    "cemetery",
    "brownfield",
    "meadow",
]
LEISURE_FLAGS = [
    "pitch",
    "park",
    "playground",
    "garden",
    "sports_centre",
    "outdoor_seating",
    "nature_reserve",
    "fitness_station",
    "golf_course",
    "swimming_pool",
]
NATURAL_FLAGS = ["wood", "water", "tree_row", "wetland", "scrub"]
SPORT_FLAGS = [
    "soccer",
    "basketball",
    "tennis",
    "multi",
    "fitness",
    "netball",
    "cricket",
    "cricket_nets",
]
ROAD_TYPES = [
    "cycleway",
    "footway",
    "primary",
    "residential",
    "secondary",
    "service",
    "tertiary",
    "unclassified",
]

GREEN_BOOLEAN_KEYS = (
    "landuse-grass?",
    "landuse-meadow?",
    "landuse-allotments?",
    "landuse-recreation-ground?",
    "leisure-park?",
    "leisure-garden?",
    "natural-wood?",
    "natural-scrub?",
    "natural-tree-row?",
)

PatchAttributes = TypedDict(
    "PatchAttributes",
    {
        "occupied?": bool,
        "hex-id": str,
        "location-id": str,
        "geo-lat": float,
        "geo-lon": float,
        "building-count": int,
        "building-density-km2": float,
        "area-size-km2": float,
        "road-cycleway": int,
        "road-footway": int,
        "road-primary": int,
        "road-residential": int,
        "road-secondary": int,
        "road-service": int,
        "road-tertiary": int,
        "road-unclassified": int,
        "total-roads": int,
        "landuse-residential?": bool,
        "landuse-grass?": bool,
        "landuse-retail?": bool,
        "landuse-construction?": bool,
        "landuse-industrial?": bool,
        "landuse-recreation-ground?": bool,
        "landuse-commercial?": bool,
        "landuse-railway?": bool,
        "landuse-allotments?": bool,
        "landuse-religious?": bool,
        "landuse-flowerbed?": bool,
        "landuse-cemetery?": bool,
        "landuse-brownfield?": bool,
        "landuse-meadow?": bool,
        "leisure-pitch?": bool,
        "leisure-park?": bool,
        "leisure-playground?": bool,
        "leisure-garden?": bool,
        "leisure-sports-centre?": bool,
        "leisure-outdoor-seating?": bool,
        "leisure-nature-reserve?": bool,
        "leisure-fitness-station?": bool,
        "leisure-golf-course?": bool,
        "leisure-swimming-pool?": bool,
        "natural-wood?": bool,
        "natural-water?": bool,
        "natural-tree-row?": bool,
        "natural-wetland?": bool,
        "natural-scrub?": bool,
        "sport-soccer?": bool,
        "sport-basketball?": bool,
        "sport-tennis?": bool,
        "sport-multi?": bool,
        "sport-fitness?": bool,
        "sport-netball?": bool,
        "sport-cricket?": bool,
        "sport-cricket-nets?": bool,
        "landuse-all": str,
        "leisure-all": str,
        "natural-all": str,
        "sport-all": str,
        "zone-type": str,
        "traversable": bool,
    },
)


@dataclass(frozen=True)
class GridBounds:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@dataclass(frozen=True)
class LoadedPatch:
    pxcor: int
    pycor: int
    attrs: PatchAttributes


def default_csv_path() -> Path:
    return Path(__file__).resolve().parent.parent / "southwark_reference_data_table.csv"


def clean_numpy_syntax(raw: str) -> str:
    """Replace numpy array(..., dtype=object) literals with plain list syntax."""
    cleaned = re.sub(r"array\(\[([^\]]*)\]\s*,\s*dtype=object\)", r"[\1]", raw, flags=re.DOTALL)
    cleaned = re.sub(r"array\(\[\]\s*,\s*dtype=object\)", "[]", cleaned)
    return cleaned


def parse_osm_json(raw: str) -> dict[str, Any]:
    """Parse OSM structured payload from CSV into a dictionary."""
    cleaned = clean_numpy_syntax(raw).replace("\n", " ")
    try:
        parsed = ast.literal_eval(cleaned)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def decompose_sports(sport_list: list[Any]) -> set[str]:
    """Split compound sport values like 'soccer;cricket' into atomic categories."""
    atoms: set[str] = set()
    for sport in sport_list:
        for atom in re.split(r"[;,]", str(sport)):
            token = atom.strip().lower()
            if token:
                atoms.add(token)
    return atoms


def load_csv_rows(csv_path: Path | str | None = None) -> list[dict[str, str]]:
    path = Path(csv_path) if csv_path is not None else default_csv_path()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def compute_grid_positions(
    rows: list[dict[str, Any]],
    ncols: int = NCOLS,
    nrows: int = NROWS,
) -> tuple[list[dict[str, Any]], GridBounds]:
    lats = [float(row["latitude"]) for row in rows]
    lons = [float(row["longitude"]) for row in rows]

    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)

    positioned: list[dict[str, Any]] = []
    seen: dict[tuple[int, int], str] = {}

    lat_span = lat_max - lat_min
    lon_span = lon_max - lon_min
    if lat_span == 0 or lon_span == 0:
        raise ValueError("Latitude/longitude range must be non-zero.")

    for row in rows:
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        pycor = round((lat - lat_min) / lat_span * (nrows - 1))
        pxcor = round((lon - lon_min) / lon_span * (ncols - 1))

        key = (pxcor, pycor)
        if key in seen:
            raise ValueError(
                f"Grid collision at {key}: {row.get('h3_index')} and {seen[key]} map to same patch"
            )
        seen[key] = str(row.get("h3_index", ""))

        updated = dict(row)
        updated["pxcor"] = pxcor
        updated["pycor"] = pycor
        positioned.append(updated)

    return positioned, GridBounds(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


def classify_zone(attrs: Mapping[str, Any]) -> str:
    density = safe_float(attrs.get("building-density-km2"), 0.0)

    has_green = any(bool(attrs.get(flag, False)) for flag in GREEN_BOOLEAN_KEYS)
    has_water = bool(attrs.get("natural-water?", False))
    has_industrial = bool(attrs.get("landuse-industrial?", False))
    has_residential = bool(attrs.get("landuse-residential?", False))
    has_commercial = bool(attrs.get("landuse-commercial?", False)) or bool(attrs.get("landuse-retail?", False))

    if has_water:
        return "water"
    if has_green and density < 1000:
        return "green-space"
    if has_green and density >= 1000:
        return "park-urban"
    if has_industrial and not has_residential:
        return "industrial"
    if density > 8000 or (has_commercial and density > 5000):
        return "super-urban"
    if has_commercial and density > 3000:
        return "commercial"
    if has_residential:
        return "residential"
    return "mixed"


def _normalized_category_set(raw_values: Any) -> set[str]:
    if not isinstance(raw_values, list):
        return set()
    return {str(value).strip().lower() for value in raw_values if str(value).strip()}


def extract_patch_attributes(row: Mapping[str, Any], water_is_barrier: bool = True) -> PatchAttributes:
    osm = parse_osm_json(str(row.get("osm_structured_json_dict", "{}")))
    summary = osm.get("summary", {}) if isinstance(osm, dict) else {}

    building = summary.get("building", {}) if isinstance(summary, dict) else {}
    roads = summary.get("roads", {}) if isinstance(summary, dict) else {}

    landuse_set = _normalized_category_set(summary.get("landuse", []))
    leisure_set = _normalized_category_set(summary.get("leisure", []))
    natural_set = _normalized_category_set(summary.get("natural", []))
    sport_atoms = decompose_sports(summary.get("sport", []) if isinstance(summary.get("sport", []), list) else [])

    attrs: dict[str, Any] = {
        "occupied?": True,
        "hex-id": str(row.get("h3_index", "")),
        "location-id": str(row.get("location_id", "")),
        "geo-lat": safe_float(row.get("latitude"), 0.0),
        "geo-lon": safe_float(row.get("longitude"), 0.0),
        "building-count": safe_int(building.get("count"), 0),
        "building-density-km2": safe_float(summary.get("building_density_km_2"), 0.0),
        "area-size-km2": safe_float(summary.get("area_size_km_2"), 0.0),
    }

    total_roads = 0
    for road_type in ROAD_TYPES:
        value = safe_int(roads.get(road_type), 0)
        attrs[f"road-{road_type}"] = value
        total_roads += value
    attrs["total-roads"] = total_roads

    for category in LANDUSE_FLAGS:
        attrs[f"landuse-{category.replace('_', '-')}?"] = category in landuse_set
    for category in LEISURE_FLAGS:
        attrs[f"leisure-{category.replace('_', '-')}?"] = category in leisure_set
    for category in NATURAL_FLAGS:
        attrs[f"natural-{category.replace('_', '-')}?"] = category in natural_set
    for category in SPORT_FLAGS:
        attrs[f"sport-{category.replace('_', '-')}?"] = category in sport_atoms

    attrs["landuse-all"] = "|".join(sorted(landuse_set)) if landuse_set else ""
    attrs["leisure-all"] = "|".join(sorted(leisure_set)) if leisure_set else ""
    attrs["natural-all"] = "|".join(sorted(natural_set)) if natural_set else ""
    attrs["sport-all"] = "|".join(sorted(sport_atoms)) if sport_atoms else ""

    attrs["zone-type"] = classify_zone(attrs)
    attrs["traversable"] = bool(attrs["occupied?"]) and (attrs["zone-type"] != "water" or not water_is_barrier)

    return attrs  # type: ignore[return-value]


def load_patch_data_from_csv(
    csv_path: Path | str | None = None,
    water_is_barrier: bool = True,
    ncols: int = NCOLS,
    nrows: int = NROWS,
) -> tuple[list[LoadedPatch], GridBounds]:
    rows = load_csv_rows(csv_path)
    positioned_rows, bounds = compute_grid_positions(rows, ncols=ncols, nrows=nrows)

    loaded: list[LoadedPatch] = []
    for row in positioned_rows:
        attrs = extract_patch_attributes(row, water_is_barrier=water_is_barrier)
        loaded.append(
            LoadedPatch(
                pxcor=int(row["pxcor"]),
                pycor=int(row["pycor"]),
                attrs=attrs,
            )
        )

    return loaded, bounds
