from __future__ import annotations

import html
import json

from musak_shared.notation.schema import ScoreData
from musak_shared.paths import STATIC_DIR

_NOTATION_JS_PATH = STATIC_DIR / "js" / "shared" / "notation.js"


def score_data_html(score_data: ScoreData, *, element_id: str = "model-output-notation") -> str:
    score_payload = html.escape(json.dumps(score_data.model_dump(mode="json")), quote=False)
    escaped_id = html.escape(element_id, quote=True)
    module_source = _NOTATION_JS_PATH.read_text(encoding="utf-8")
    return f"""
    <!doctype html>
    <html>
    <head>
      <style>
        :root {{
          color-scheme: light dark;
        }}

        html,
        body {{
          background: transparent;
          color: #111827;
          margin: 0;
          font-family: sans-serif;
        }}

        @media (prefers-color-scheme: dark) {{
          body {{
            color: #f9fafb;
          }}
        }}

        #{escaped_id} {{
          background: transparent;
        }}

        #{escaped_id} svg {{
          background: transparent !important;
          color: inherit;
        }}

        #{escaped_id} svg rect {{
          fill: transparent !important;
        }}

        #{escaped_id} svg path,
        #{escaped_id} svg line,
        #{escaped_id} svg text {{
          fill: currentColor !important;
          stroke: currentColor !important;
        }}
      </style>
    </head>
    <body>
    <div id="{escaped_id}" style="width: 100%; overflow-x: auto;">Loading notation...</div>
    <script type="module">
      const container = document.getElementById({json.dumps(element_id)});
      try {{
        const moduleSource = {json.dumps(module_source)};
        const moduleUrl = URL.createObjectURL(new Blob([moduleSource], {{ type: 'text/javascript' }}));
        const {{ renderScore }} = await import(moduleUrl);
        const scoreData = JSON.parse({json.dumps(score_payload)});
        renderScore(scoreData, container);
        URL.revokeObjectURL(moduleUrl);
      }} catch (error) {{
        container.textContent = `Notation render failed: ${{error.message}}`;
      }}
    </script>
    </body>
    </html>
    """
