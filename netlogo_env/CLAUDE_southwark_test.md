# Project Fram

Hackathon project converting Southwark (London) geospatial data into a NetLogo agent-based simulation environment.

## Data Pipeline

1. **Source**: `southwark_reference_data_table.csv` — 364 H3 resolution-10 hexagons with OSM features and satellite image descriptions
2. **Generator**: `generate_netlogo_model.py` — Python script that parses the CSV and produces a self-contained `.nlogo` file
3. **Output**: `southwark_environment.nlogo` — NetLogo model with all patch attributes and giraffe agents

## Key Files

- `southwark_reference_data_table.csv` — raw data (364 rows, columns: h3_index, description, location_id, osm_structured_json_dict, latitude, longitude)
- `generate_netlogo_model.py` — run with `python3 generate_netlogo_model.py` to regenerate the .nlogo file
- `southwark_environment.nlogo` — open directly in NetLogo 6.4.0, click Setup then Go
- `Example.ipynb` — reference notebook showing eikonsai API usage and H3 grid operations

## Data Format Notes

- The `osm_structured_json_dict` column uses Python repr with numpy `array(... dtype=object)` syntax — the generator handles this via regex cleanup + `ast.literal_eval`
- Geographic coverage: Southwark borough, lat 51.42–51.51, lon -0.11 to -0.03
- Grid mapping: 26 columns (longitude) x 34 rows (latitude), ~15px patches in NetLogo

## NetLogo Model

- 58 patch variables (buildings, roads, landuse/leisure/natural/sport booleans, full category strings, zone-type classification)
- Zone classifications: water, green-space, park-urban, industrial, super-urban, commercial, residential, mixed
- Giraffe agents with energy mechanics (deplete on move, replenish on green patches) and distance tracking
- 6 visualization modes: zone-type (default), building-density, landuse-dominant, road-density, natural-features, sport-facilities
- 9 monitors (giraffes alive, mean energy, green/water/super-urban/residential patch counts, total buildings, avg distance, giraffes on green)
- Population & Energy plot tracking trends over time
