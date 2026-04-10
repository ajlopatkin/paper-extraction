from __future__ import annotations

BOX_PLOT_SYSTEM = """You extract numeric data from scientific figure panels. Reply with a single JSON object only — no markdown fences, no commentary. Use null for unreadable values."""

BOX_PLOT_USER = """Extract all data from this box plot panel. For each group/condition shown:
1. Read the group label (x-axis label or legend entry)
2. Read the median value (center line of box)
3. Read the lower and upper quartile (box edges)
4. Read the whisker extents (min/max)
5. Note any statistical annotations (*, **, ***, ns, P-values) as a short string in "significance"
6. Read the axis labels and y-axis units if visible

Return exactly this JSON shape:
{{
  "plot_type": "box_plot",
  "axis_x_label": string or null,
  "axis_y_label": string or null,
  "axis_y_units": string or null,
  "groups": [
    {{
      "label": string,
      "median": number or null,
      "q1": number or null,
      "q3": number or null,
      "whisker_low": number or null,
      "whisker_high": number or null,
      "significance": string or null
    }}
  ]
}}"""

LINE_CHART_SYSTEM = """You extract numeric data from scientific figure panels. Reply with a single JSON object only — no markdown fences, no commentary. Use null for unreadable values."""

LINE_CHART_USER = """Extract all data from this line or scatter plot panel. For each series (line), read series name from legend if present. For each visible point, read x and y; include error bar low/high if shown.

Return exactly this JSON shape (use "plot_type": "line_chart" or "line_plot" — same schema):
{{
  "plot_type": "line_chart",
  "axis_x_label": string or null,
  "axis_y_label": string or null,
  "axis_x_units": string or null,
  "axis_y_units": string or null,
  "series": [
    {{
      "name": string or null,
      "points": [
        {{ "x": number or string, "y": number, "error_low": number or null, "error_high": number or null }}
      ]
    }}
  ]
}}"""

TABLE_IMAGE_SYSTEM = """You transcribe tables shown as images in scientific papers. Reply with a single JSON object only — no markdown fences, no commentary. Use empty string \"\" for blank cells."""

TABLE_IMAGE_USER = """This image is a table (data printed as a graphic, not selectable text). Transcribe it faithfully.

Rules:
1. If there is a clear header row, put those labels in "column_headers" (left to right) and put **data only** in "rows".
2. If headers are ambiguous or merged, put every visible row (including a header-like row) in "rows" and leave "column_headers" empty.
3. Each data row is an array of cell strings, same length as other rows when possible; pad with \"\" if a cell is missing.
4. Preserve units and symbols as text (e.g. \"μg/mL\", \"10³\").
5. If part of the table is unreadable, put a short explanation in "notes".

Return exactly this JSON shape:
{{
  "plot_type": "table_image",
  "title_or_caption": string or null,
  "column_headers": [ string, ... ],
  "rows": [ [ string, ... ], ... ],
  "notes": string or null
}}"""

PLASMID_MAP_SYSTEM = """You describe molecular biology vector / plasmid diagrams from scientific figures. Reply with a single JSON object only — no markdown fences, no commentary. Use null or empty arrays where unknown."""

PLASMID_MAP_USER = """This image is a plasmid map, cloning vector, or similar circular/linear DNA diagram (not a data chart). Extract what is legible.

1. If a plasmid name or construct ID is visible, put it in "map_name_or_identifier".
2. Set "is_circular" to true if drawn as a circle, false if linear-only, null if unclear.
3. For each labeled feature (genes, promoters, origins, resistance cassettes, MCS, restriction sites with names, etc.), add an object to "features" with "label" (exact or best reading), optional "feature_type", optional "notes".
4. Put leftover readable text (sizes in bp, misc labels) in "other_visible_labels" as strings.
5. Short caveats (blur, overlap) go in "notes".

Return exactly this JSON shape:
{{
  "plot_type": "plasmid_map",
  "title_or_caption": string or null,
  "map_name_or_identifier": string or null,
  "is_circular": boolean or null,
  "features": [
    {{ "label": string, "feature_type": string or null, "notes": string or null }}
  ],
  "other_visible_labels": [ string, ... ],
  "notes": string or null
}}"""

WORKFLOW_DIAGRAM_SYSTEM = """You describe workflow, flowchart, and pipeline diagrams from scientific figures. Reply with a single JSON object only — no markdown fences, no commentary. Use null or empty arrays where unknown."""

WORKFLOW_DIAGRAM_USER = """This image is a workflow diagram, flowchart, pipeline schematic, or similar (boxes and arrows / connectors — not a numeric plot). Extract what is legible.

1. For each distinct box, rounded rectangle, diamond, circle, or labeled region, add an object to "nodes" with "label" (main text), optional "node_type" (start_end, process, decision, data, document, subprocess, other), optional "notes".
2. For each visible flow from one node to another, add to "edges" with "from_label" and "to_label" matching the node labels (best effort if text wraps). Optional "edge_label" for Yes/No, conditions, or arrow text.
3. Put leftover readable text (legends, footnotes) in "other_visible_labels".
4. Short caveats in "notes".

Return exactly this JSON shape:
{{
  "plot_type": "workflow_diagram",
  "title_or_caption": string or null,
  "nodes": [
    {{ "label": string, "node_type": string or null, "notes": string or null }}
  ],
  "edges": [
    {{ "from_label": string, "to_label": string, "edge_label": string or null }}
  ],
  "other_visible_labels": [ string, ... ],
  "notes": string or null
}}"""

EXPERIMENTAL_WORKFLOW_SYSTEM = """You describe experimental and laboratory protocol diagrams from scientific figures. Reply with a single JSON object only — no markdown fences, no commentary. Use null or empty arrays where unknown."""

EXPERIMENTAL_WORKFLOW_USER = """This image is an experimental workflow, lab protocol schematic, or study-design pipeline (e.g. animal model → treatment → sampling → assay; cell culture steps; clinical trial arms). It is **not** a numeric data chart.

1. For each distinct step box, stage, arm, or labeled process, add to "nodes" with "label" (main text), optional "node_type" (e.g. culture, treatment, wash, harvest, assay, imaging, randomization, inclusion), optional "notes" (timepoints, doses, temperatures if visible).
2. For each visible flow or sequence link, add to "edges" with "from_label" and "to_label" matching node labels. Optional "edge_label" for conditions (e.g. \"control\", \"Day 7\", \"if positive\").
3. Put legends, scale bars of protocol only, or stray text in "other_visible_labels".
4. Caveats (blur, crowded panel) in "notes".

Return exactly this JSON shape:
{{
  "plot_type": "experimental_workflow",
  "title_or_caption": string or null,
  "nodes": [
    {{ "label": string, "node_type": string or null, "notes": string or null }}
  ],
  "edges": [
    {{ "from_label": string, "to_label": string, "edge_label": string or null }}
  ],
  "other_visible_labels": [ string, ... ],
  "notes": string or null
}}"""

JSON_FIX_SUFFIX = "\n\nYour previous reply was not valid JSON for the schema. Return only one corrected JSON object."