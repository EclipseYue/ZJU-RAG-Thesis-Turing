# Wikidata-Style Graph QA Dataset

This directory stores a static JSONL graph-QA dataset for Route A graph smoke tests.

`validation.jsonl` is converted from the local `2wikimultihopqa/validation.json` evidence triples. It is not an online Neo4j dependency and does not require a live Wikidata service. The `source` field is therefore set to `2wikimultihopqa_wikidata_style` to avoid over-claiming provenance while preserving a Wikidata-style graph format.

Each line has the following schema:

```json
{
  "id": "graph_001",
  "question": "...",
  "answer": "...",
  "triples": [
    {
      "head": "...",
      "relation": "...",
      "tail": "...",
      "source": "2wikimultihopqa_wikidata_style",
      "qid": "Q..."
    }
  ],
  "supporting_triples": [0]
}
```

Run with:

```bash
.venv/bin/python experiments/run_route_a_baseline.py \
  --preset experiments/presets/route_a_wikidata_graph.json \
  --samples 50 \
  --output-name route_a_wikidata_graph_smoke_50.json
```
