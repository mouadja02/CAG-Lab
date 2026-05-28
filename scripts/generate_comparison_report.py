#!/usr/bin/env python3
"""Generate a comparison report (HTML + Markdown) from two experiment JSONL files.

Usage:
    # Auto-detect latest two JSONL files and output to docs/
    python scripts/generate_comparison_report.py --auto

    # Explicit paths
    python scripts/generate_comparison_report.py \\
        results/jsonl/rag_baseline_YYYYMMDD.jsonl --label-a "RAG Baseline" \\
        results/jsonl/semantic_cache_YYYYMMDD.jsonl --label-b "Semantic Cache"

Output:
    docs/report.html  (or --output-dir/docs/report.html)
    docs/report.md
"""

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Data loading & computation
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stats(values: list[float]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p50": 0, "p95": 0}
    s = sorted(values)
    n = len(s)

    def pct(p):
        k = (p / 100) * (n - 1)
        f, c = math.floor(k), math.ceil(k)
        return s[f] if f == c else s[f] * (c - k) + s[c] * (k - f)

    return {
        "min": s[0],
        "max": s[-1],
        "mean": sum(s) / n,
        "median": pct(50),
        "p50": pct(50),
        "p95": pct(95),
    }


def _difficulty_breakdown(records: list[dict]) -> dict:
    """Breakdown of scores by difficulty level."""
    result = {}
    for diff in ["easy", "medium", "hard"]:
        group = [r for r in records if r.get("difficulty") == diff]
        if group:
            result[diff] = {
                "count": len(group),
                "mean_score": sum(r.get("judge_score", 0) for r in group) / len(group),
            }
    return result


def _retrieval_relevance_summary(records: list[dict]) -> dict:
    """Summary of retrieval relevance scores (if present)."""
    rel_scores = [
        r["retrieval_relevance"] for r in records if "retrieval_relevance" in r
    ]
    if not rel_scores:
        return {}
    return {
        "mean": sum(rel_scores) / len(rel_scores),
        "fully_relevant": sum(1 for s in rel_scores if s == 2),
        "partially_relevant": sum(1 for s in rel_scores if s == 1),
        "irrelevant": sum(1 for s in rel_scores if s == 0),
        "total": len(rel_scores),
    }


def compute_summary(records: list[dict]) -> dict:
    n = len(records)
    latencies = [r["latency_s"] for r in records]
    scores = [r.get("judge_score", 0) for r in records]
    costs = [r.get("cost", 0) for r in records]
    has_cite = sum(1 for r in records if r.get("has_citation"))

    prompt_t = sum(r.get("prompt_tokens", 0) for r in records)
    comp_t = sum(r.get("completion_tokens", 0) for r in records)

    qtypes = Counter(r.get("query_type", "?") for r in records)
    qtype_scores = {}
    for qt in qtypes:
        group = [r.get("judge_score", 0) for r in records if r.get("query_type") == qt]
        qtype_scores[qt] = {
            "count": len(group),
            "mean_score": sum(group) / len(group) if group else 0,
        }

    is_cache = any(r.get("cache_hit") is not None for r in records)
    cache_summary = {}
    if is_cache:
        hits = [r for r in records if r.get("cache_hit")]
        hits_n = len(hits)
        fp = sum(1 for r in hits if r.get("false_positive"))
        saved = sum(r.get("cost_saved", 0) for r in records)
        rels = Counter(r.get("relationship", "?") for r in records)
        rel_hits = {}
        for rel in rels:
            g = [r for r in records if r.get("relationship") == rel]
            rel_hits[rel] = {
                "count": len(g),
                "hit_count": sum(1 for r in g if r.get("cache_hit")),
            }
        miss_latencies = [r["latency_s"] for r in records if not r.get("cache_hit")]
        hit_latencies = [r["latency_s"] for r in records if r.get("cache_hit")]
        miss_costs = [r.get("cost", 0) for r in records if not r.get("cache_hit")]
        hit_costs = [r.get("cost", 0) for r in records if r.get("cache_hit")]

        # Paraphrase tier breakdown
        tier_data = {}
        tier_records = [r for r in records if r.get("paraphrase_tier")]
        if tier_records:
            for tier in ["easy", "medium", "hard"]:
                group = [r for r in tier_records if r.get("paraphrase_tier") == tier]
                if group:
                    tier_data[tier] = {
                        "count": len(group),
                        "hit_count": sum(1 for r in group if r.get("cache_hit")),
                        "mean_score": sum(r.get("judge_score", 0) for r in group)
                        / len(group),
                    }

        # False positive details
        fp_details = []
        for r in records:
            if r.get("false_positive"):
                fp_details.append(
                    {
                        "question": r.get("question", "")[:80],
                        "cached_source": r.get("cached_question_id", "?"),
                        "expected_source": r.get("question_id", "?"),
                        "match_score": r.get("cache_match_score", "?"),
                    }
                )

        cache_summary = {
            "hit_rate": hits_n / n if n else 0,
            "hit_count": hits_n,
            "miss_count": n - hits_n,
            "false_positive_rate": fp / hits_n if hits_n else 0,
            "false_positive_count": fp,
            "false_positive_details": fp_details,
            "cost_saved": saved,
            "hit_latency_stats": stats(hit_latencies),
            "miss_latency_stats": stats(miss_latencies),
            "avg_hit_cost": sum(hit_costs) / len(hit_costs) if hit_costs else 0,
            "avg_miss_cost": sum(miss_costs) / len(miss_costs) if miss_costs else 0,
            "by_relationship": rel_hits,
            "by_paraphrase_tier": tier_data,
        }

    return {
        "n": n,
        "latency_stats": stats(latencies),
        "judge_stats": stats([float(s) for s in scores]),
        "mean_score": sum(scores) / n if n else 0,
        "citation_rate": has_cite / n if n else 0,
        "total_prompt_tokens": prompt_t,
        "total_completion_tokens": comp_t,
        "total_tokens": prompt_t + comp_t,
        "total_cost": sum(costs),
        "cost_per_1k": (sum(costs) / n * 1000) if n else 0,
        "latencies": latencies,
        "scores": scores,
        "costs": costs,
        "qtype_scores": qtype_scores,
        "cache": cache_summary,
        "difficulty_scores": _difficulty_breakdown(records),
        "retrieval_relevance": _retrieval_relevance_summary(records),
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _json_dumps(obj):
    return json.dumps(obj, ensure_ascii=False)


def _detect_model(report_path: Path) -> str:
    """Extract model name from a markdown report title like '# RAG Baseline — model @ top-k'."""
    try:
        with open(report_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("# ") and " — " in line:
                    return line.split(" — ")[1].split(" @")[0].strip()
    except Exception:
        pass
    return "unknown-model"


def generate_html(
    label_a: str,
    summary_a: dict,
    is_cache_a: bool,
    label_b: str,
    summary_b: dict,
    is_cache_b: bool,
    output_path: str,
    model: str = "",
) -> str:
    qt_a = _json_dumps(summary_a["qtype_scores"])
    qt_b = _json_dumps(summary_b["qtype_scores"])
    lat_a = _json_dumps(summary_a["latencies"])
    lat_b = _json_dumps(summary_b["latencies"])
    cache_a = _json_dumps(summary_a["cache"])
    cache_b = _json_dumps(summary_b["cache"])

    rel_data = "null"
    if summary_b.get("cache"):
        rel_data = _json_dumps(summary_b["cache"]["by_relationship"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CAG-Lab Benchmark Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; background:#0b0f19; color:#e2e8f0; line-height:1.6; }}
.container {{ max-width:1200px; margin:0 auto; padding:40px 24px; }}
header {{ text-align:center; margin-bottom:48px; }}
header h1 {{ font-size:2.2rem; font-weight:700; background:linear-gradient(135deg,#818cf8,#c084fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
header p {{ color:#94a3b8; margin-top:8px; font-size:0.95rem; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px; margin-bottom:48px; }}
.card {{ background:#1e293b; border-radius:12px; padding:24px; border:1px solid #334155; }}
.card .label {{ font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px; }}
.card .value {{ font-size:2rem; font-weight:700; }}
.card .value.green {{ color:#34d399; }}
.card .value.blue {{ color:#60a5fa; }}
.card .value.purple {{ color:#a78bfa; }}
.card .value.amber {{ color:#fbbf24; }}
.card .sub {{ font-size:0.8rem; color:#64748b; margin-top:4px; }}
.section {{ margin-bottom:48px; }}
.section h2 {{ font-size:1.4rem; font-weight:600; margin-bottom:20px; color:#e2e8f0; border-bottom:1px solid #334155; padding-bottom:8px; }}
.chart-row {{ display:grid; grid-template-columns:1fr 1fr; gap:24px; }}
@media(max-width:768px) {{ .chart-row {{ grid-template-columns:1fr; }} }}
.chart-box {{ background:#1e293b; border-radius:12px; padding:24px; border:1px solid #334155; }}
.chart-box h3 {{ font-size:0.95rem; color:#cbd5e1; margin-bottom:16px; text-align:center; }}
.chart-box canvas {{ max-height:320px; }}
table {{ width:100%; border-collapse:collapse; background:#1e293b; border-radius:12px; overflow:hidden; border:1px solid #334155; }}
th, td {{ padding:12px 16px; text-align:left; }}
th {{ background:#0f172a; color:#94a3b8; font-weight:600; font-size:0.85rem; text-transform:uppercase; }}
td {{ border-top:1px solid #334155; font-size:0.9rem; }}
tr:hover td {{ background:#263348; }}
.table-compare td:first-child {{ color:#94a3b8; width:40%; }}
.table-compare td:nth-child(2), .table-compare td:nth-child(3) {{ font-weight:600; }}
.highlight {{ color:#34d399; }}
.dim {{ color:#f87171; }}
footer {{ text-align:center; color:#475569; font-size:0.8rem; margin-top:48px; padding-top:24px; border-top:1px solid #1e293b; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:6px; font-size:0.75rem; font-weight:600; }}
.badge.green {{ background:#064e3b; color:#34d399; }}
.badge.blue {{ background:#1e3a5f; color:#60a5fa; }}
.badge.purple {{ background:#3b0764; color:#a78bfa; }}
.full-width {{ grid-column:1/-1; }}
.top-nav {{ position:sticky; top:0; z-index:100; background:rgba(11,15,25,.92); backdrop-filter:blur(12px); border-bottom:1px solid #334155; }}
.top-nav-inner {{ max-width:1200px; margin:0 auto; padding:0 24px; display:flex; align-items:center; gap:24px; height:48px; }}
.top-nav-inner a {{ color:#94a3b8; text-decoration:none; font-size:0.85rem; font-weight:500; }}
.top-nav-inner a:hover {{ color:#e2e8f0; }}
.top-nav-inner .brand {{ font-size:1.05rem; font-weight:700; background:linear-gradient(135deg,#818cf8,#c084fc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
</style>
</head>
<body>
<div class="top-nav"><div class="top-nav-inner">
  <a href="index.html" class="brand">CAG-Lab</a>
  <a href="index.html">Overview</a>
  <a href="rag.html">RAG</a>
  <a href="cag.html">CAG</a>
  <a href="report.html" style="color:#e2e8f0;">Report</a>
</div></div>
<div class="container">
<header>
  <h1>CAG-Lab Benchmark Comparison</h1>
  <p>{label_a} vs {label_b} — {model or "see config"}</p>
  <p style="font-size:0.8rem; color:#64748b;">Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</p>
</header>

<div class="cards">
  <div class="card">
    <div class="label">Mean Judge Score</div>
    <div class="value green">{summary_a["mean_score"]:.3f} → {summary_b["mean_score"]:.3f}</div>
    <div class="sub">{"%+.0f%%" % ((summary_b["mean_score"] - summary_a["mean_score"]) / summary_a["mean_score"] * 100) if summary_a["mean_score"] else ""} change</div>
  </div>
  <div class="card">
    <div class="label">p50 Latency</div>
    <div class="value blue">{summary_a["latency_stats"]["p50"]:.2f}s → {summary_b["latency_stats"]["p50"]:.2f}s</div>
    <div class="sub">Cache: {((1 - summary_b["latency_stats"]["p50"] / (summary_a["latency_stats"]["p50"] if summary_a["latency_stats"]["p50"] else 1)) * 100):.0f}% faster</div>
  </div>
  <div class="card">
    <div class="label">Cost per 1k Questions</div>
    <div class="value purple">${summary_a["cost_per_1k"]:.4f} → ${summary_b["cost_per_1k"]:.4f}</div>
    <div class="sub">{"%+.0f%%" % ((summary_b["cost_per_1k"] - summary_a["cost_per_1k"]) / summary_a["cost_per_1k"] * 100) if summary_a["cost_per_1k"] else ""} change</div>
  </div>
  <div class="card">
    <div class="label">Citation Rate</div>
    <div class="value amber">{summary_a["citation_rate"]:.1%} → {summary_b["citation_rate"]:.1%}</div>
    <div class="sub">{"%+.0fpp" % ((summary_b["citation_rate"] - summary_a["citation_rate"]) * 100)} change</div>
  </div>
</div>
"""

    # Second row of cards for cache metrics
    if summary_b.get("cache"):
        c = summary_b["cache"]
        html += f"""
<div class="cards">
  <div class="card">
    <div class="label">Cache Hit Rate</div>
    <div class="value green">{c["hit_rate"]:.1%}</div>
    <div class="sub">{c["hit_count"]} hits / {summary_b["n"]} total</div>
  </div>
  <div class="card">
    <div class="label">False-Positive Rate</div>
    <div class="value" style="color:#f87171">{c["false_positive_rate"]:.1%}</div>
    <div class="sub">{c["false_positive_count"]} of {c["hit_count"]} hits were wrong source</div>
  </div>
  <div class="card">
    <div class="label">Cost Saved</div>
    <div class="value purple">${c["cost_saved"]:.6f}</div>
    <div class="sub">Generation cost avoided</div>
  </div>
  <div class="card">
    <div class="label">Hit vs Miss Latency</div>
    <div class="value blue">{c["hit_latency_stats"]["p50"]:.2f}s vs {c["miss_latency_stats"]["p50"]:.2f}s</div>
    <div class="sub">p50: cache hits are {((1 - c["hit_latency_stats"]["p50"] / (c["miss_latency_stats"]["p50"] if c["miss_latency_stats"]["p50"] else 1)) * 100):.0f}% faster</div>
  </div>
</div>
"""

    # Charts section
    html += """
<div class="section">
  <h2>Performance Comparison</h2>
  <div class="chart-row">
    <div class="chart-box">
      <h3>Latency (p50 / p95)</h3>
      <canvas id="chartLatency"></canvas>
    </div>
    <div class="chart-box">
      <h3>Cost Comparison</h3>
      <canvas id="chartCost"></canvas>
    </div>
  </div>
  <div class="chart-row" style="margin-top:24px;">
    <div class="chart-box">
      <h3>Latency Distribution</h3>
      <canvas id="chartLatDist"></canvas>
    </div>
    <div class="chart-box">
      <h3>Judge Score by Query Type</h3>
      <canvas id="chartQtypeScore"></canvas>
    </div>
  </div>
"""

    if summary_b.get("cache"):
        html += """
  <div class="chart-row" style="margin-top:24px;">
    <div class="chart-box">
      <h3>Cache Hit Rate by Relationship</h3>
      <canvas id="chartRelHit"></canvas>
    </div>
    <div class="chart-box">
      <h3>Cost Breakdown</h3>
      <canvas id="chartCostPie"></canvas>
    </div>
  </div>
"""

    html += """
</div>
"""

    # Table section
    html += (
        """
<div class="section">
  <h2>Query Type Breakdown</h2>
  <table class="table-compare">
    <thead>
      <tr>
        <th>Query Type</th>
        <th style="text-align:center">"""
        + label_a
        + """ Score</th>
        <th style="text-align:center">"""
        + label_b
        + """ Score</th>
        <th style="text-align:center">Δ</th>
      </tr>
    </thead>
    <tbody>
"""
    )
    all_qtypes = sorted(set(summary_a["qtype_scores"]) | set(summary_b["qtype_scores"]))
    for qt in all_qtypes:
        sa = summary_a["qtype_scores"].get(qt, {}).get("mean_score", 0)
        sb = summary_b["qtype_scores"].get(qt, {}).get("mean_score", 0)
        diff = sb - sa
        cls = "highlight" if diff > 0 else ("dim" if diff < 0 else "")
        na = summary_a["qtype_scores"].get(qt, {}).get("count", 0)
        nb = summary_b["qtype_scores"].get(qt, {}).get("count", 0)
        html += f"""      <tr>
        <td>{qt} <span style="color:#64748b;font-size:0.75rem;">(n={na}/{nb})</span></td>
        <td style="text-align:center">{sa:.3f}</td>
        <td style="text-align:center">{sb:.3f}</td>
        <td style="text-align:center" class="{cls}">{diff:+.3f}</td>
      </tr>
"""
    html += """    </tbody>
  </table>
</div>
"""

    # Token/cost table
    html += f"""
<div class="section">
  <h2>Detailed Metrics</h2>
  <table class="table-compare">
    <thead>
      <tr><th>Metric</th><th style="text-align:center">{label_a}</th><th style="text-align:center">{label_b}</th></tr>
    </thead>
    <tbody>
      <tr><td>Questions</td><td style="text-align:center">{summary_a["n"]}</td><td style="text-align:center">{summary_b["n"]}</td></tr>
      <tr><td>Mean Judge Score</td><td style="text-align:center">{summary_a["mean_score"]:.3f}</td><td style="text-align:center">{summary_b["mean_score"]:.3f}</td></tr>
      <tr><td>Citation Rate</td><td style="text-align:center">{summary_a["citation_rate"]:.1%}</td><td style="text-align:center">{summary_b["citation_rate"]:.1%}</td></tr>
      <tr><td>p50 Latency</td><td style="text-align:center">{summary_a["latency_stats"]["p50"]:.2f} s</td><td style="text-align:center">{summary_b["latency_stats"]["p50"]:.2f} s</td></tr>
      <tr><td>p95 Latency</td><td style="text-align:center">{summary_a["latency_stats"]["p95"]:.2f} s</td><td style="text-align:center">{summary_b["latency_stats"]["p95"]:.2f} s</td></tr>
      <tr><td>Total Tokens</td><td style="text-align:center">{summary_a["total_tokens"]:,}</td><td style="text-align:center">{summary_b["total_tokens"]:,}</td></tr>
      <tr><td>Total Cost</td><td style="text-align:center">${summary_a["total_cost"]:.6f}</td><td style="text-align:center">${summary_b["total_cost"]:.6f}</td></tr>
      <tr><td>Cost per 1k Questions</td><td style="text-align:center">${summary_a["cost_per_1k"]:.6f}</td><td style="text-align:center">${summary_b["cost_per_1k"]:.6f}</td></tr>
    </tbody>
  </table>
</div>
"""

    # JavaScript for charts
    html += f"""
<script>
const LATA = "{label_a}";
const LATB = "{label_b}";
const qtypeScoresA = {qt_a};
const qtypeScoresB = {qt_b};
const latA = {lat_a};
const latB = {lat_b};

Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';
Chart.defaults.font.family = 'Inter';

// Adaptive bin size: use p95 of combined latencies to avoid long-tail distortion
const _allLat = [...latA, ...latB].filter(v => v > 0).sort((a, b) => a - b);
const _p95idx = Math.max(0, Math.floor(_allLat.length * 0.95) - 1);
const _latMax = (_allLat[_p95idx] || Math.max(..._allLat, 1));
const _rawBin = _latMax / 20;
const _mag = Math.pow(10, Math.floor(Math.log10(_rawBin || 1)));
const binSize = Math.max(0.5, Math.ceil(_rawBin / _mag) * _mag);
const latEnd = Math.ceil(_latMax / binSize) * binSize;

function makeBins(data, end, bSize) {{
  if (!data.length) return {{labels:[],bins:[]}};
  const n = Math.ceil(end / bSize);
  const labels = [];
  const bins = new Array(n).fill(0);
  for (let i = 0; i < n; i++) labels.push((i*bSize).toFixed(1)+'-'+((i+1)*bSize).toFixed(1)+'s');
  for (const v of data) {{
    if (v > end) continue;
    bins[Math.min(Math.floor(v / bSize), n-1)]++;
  }}
  return {{labels, bins}};
}}

const ctx1 = document.getElementById('chartLatency');
new Chart(ctx1, {{
  type: 'bar',
  data: {{
    labels: ['p50','p95'],
    datasets: [
      {{ label: LATA, data: [{summary_a["latency_stats"]["p50"]:.3f},{summary_a["latency_stats"]["p95"]:.3f}], backgroundColor: '#60a5fa80', borderColor: '#60a5fa', borderWidth: 1 }},
      {{ label: LATB, data: [{summary_b["latency_stats"]["p50"]:.3f},{summary_b["latency_stats"]["p95"]:.3f}], backgroundColor: '#a78bfa80', borderColor: '#a78bfa', borderWidth: 1 }},
    ]
  }},
  options: {{ responsive:true, plugins:{{ legend:{{position:'bottom',labels:{{padding:20}}}} }}, scales:{{ y:{{ title:{{display:true,text:'Seconds'}}, beginAtZero:true }} }} }}
}});

const ctx2 = document.getElementById('chartCost');
new Chart(ctx2, {{
  type: 'bar',
  data: {{
    labels: ['Total Cost ($)','Cost/1k Q ($)'],
    datasets: [
      {{ label: LATA, data: [{summary_a["total_cost"]:.6f},{summary_a["cost_per_1k"]:.6f}], backgroundColor: '#60a5fa80', borderColor: '#60a5fa', borderWidth: 1 }},
      {{ label: LATB, data: [{summary_b["total_cost"]:.6f},{summary_b["cost_per_1k"]:.6f}], backgroundColor: '#a78bfa80', borderColor: '#a78bfa', borderWidth: 1 }},
    ]
  }},
  options: {{ responsive:true, plugins:{{ legend:{{position:'bottom',labels:{{padding:20}}}} }}, scales:{{ y:{{ beginAtZero:true }} }} }}
}});

const binsA = makeBins(latA, latEnd, binSize);
const binsB = makeBins(latB, latEnd, binSize);
const ctx3 = document.getElementById('chartLatDist');
new Chart(ctx3, {{
  type: 'bar',
  data: {{
    labels: binsA.labels,
    datasets: [
      {{ label: LATA, data: binsA.bins, backgroundColor: '#60a5fa60', borderColor: '#60a5fa', borderWidth: 1 }},
      {{ label: LATB, data: binsB.bins.concat(new Array(Math.max(0, binsA.bins.length - binsB.bins.length)).fill(0)), backgroundColor: '#a78bfa60', borderColor: '#a78bfa', borderWidth: 1 }},
    ]
  }},
  options: {{ responsive:true, plugins:{{ legend:{{position:'bottom',labels:{{padding:20}}}} }}, scales:{{ x:{{ title:{{display:true,text:'Latency (seconds)'}} }}, y:{{ title:{{display:true,text:'Questions'}}, beginAtZero:true }} }} }}
}});

const qtypes = Object.keys(qtypeScoresA).sort();
const ctx4 = document.getElementById('chartQtypeScore');
new Chart(ctx4, {{
  type: 'bar',
  data: {{
    labels: qtypes,
    datasets: [
      {{ label: LATA, data: qtypes.map(q=>qtypeScoresA[q]?.mean_score||0), backgroundColor: '#60a5fa80', borderColor: '#60a5fa', borderWidth: 1 }},
      {{ label: LATB, data: qtypes.map(q=>(qtypeScoresB[q]||qtypeScoresA[q])?.mean_score||0), backgroundColor: '#a78bfa80', borderColor: '#a78bfa', borderWidth: 1 }},
    ]
  }},
  options: {{ responsive:true, plugins:{{ legend:{{position:'bottom',labels:{{padding:20}}}} }}, scales:{{ y:{{ min:0, max:1, title:{{display:true,text:'Mean Judge Score'}} }} }} }}
}});
"""

    if summary_b.get("cache"):
        html += f"""
const relData = {rel_data};
const relCtx = document.getElementById('chartRelHit');
new Chart(relCtx, {{
  type: 'bar',
  data: {{
    labels: Object.keys(relData).sort(),
    datasets: [
      {{ label: 'Count', data: Object.keys(relData).sort().map(k=>relData[k].count), backgroundColor: '#334155', borderColor: '#475569', borderWidth: 1 }},
      {{ label: 'Hits', data: Object.keys(relData).sort().map(k=>relData[k].hit_count), backgroundColor: '#34d39980', borderColor: '#34d399', borderWidth: 1 }},
    ]
  }},
  options: {{ responsive:true, plugins:{{ legend:{{position:'bottom',labels:{{padding:20}}}} }}, scales:{{ y:{{ title:{{display:true,text:'Items'}}, beginAtZero:true }} }} }}
}});

const ctxPie = document.getElementById('chartCostPie');
new Chart(ctxPie, {{
  type: 'doughnut',
  data: {{
    labels: ['Spent (generation)','Saved by cache'],
    datasets: [{{
      data: [{summary_b["total_cost"]:.6f},{summary_b["cache"]["cost_saved"]:.6f}],
      backgroundColor: ['#60a5fa','#34d399'],
      borderColor: '#0b0f19',
      borderWidth: 2,
    }}]
  }},
  options: {{ responsive:true, plugins:{{ legend:{{position:'bottom',labels:{{padding:20}}}} }} }}
}});
"""

    html += f"""
</script>

<div class="section" style="text-align:center; padding-top: 16px;">
  <a href="report.md" download class="btn primary" style="font-size:1rem; padding: 14px 32px;">
    &#8681; Download Markdown Report
  </a>
</div>

<footer>
  CAG-Lab Benchmark Report &middot; {model or "see config"}
</footer>
</div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def generate_markdown(
    label_a: str,
    summary_a: dict,
    label_b: str,
    summary_b: dict,
    output_path: str,
) -> str:
    lines = [
        f"# CAG-Lab Benchmark Comparison: {label_a} vs {label_b}",
        "",
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "## Summary",
        "",
        f"| Metric | {label_a} | {label_b} | Change |",
        f"|--------|----------|----------|--------|",
    ]

    def row(metric, va, vb, fmt=".3f", suffix="", pct=False):
        if pct:
            sa = f"{va:.1%}"
            sb = f"{vb:.1%}"
        elif fmt == ",d":
            sa = f"{va:,}"
            sb = f"{vb:,}"
        else:
            sa = f"{va:{fmt}}{suffix}"
            sb = f"{vb:{fmt}}{suffix}"
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va:
            ch = (vb - va) / va * 100
            sc = f"{ch:+.0f}%"
        else:
            sc = "-"
        return f"| {metric} | {sa} | {sb} | {sc} |"

    lines.append(row("Questions", summary_a["n"], summary_b["n"], fmt=",d"))
    lines.append(
        row("Mean Judge Score", summary_a["mean_score"], summary_b["mean_score"])
    )
    lines.append(
        row(
            "Citation Rate",
            summary_a["citation_rate"],
            summary_b["citation_rate"],
            pct=True,
        )
    )
    lines.append(
        row(
            "p50 Latency",
            summary_a["latency_stats"]["p50"],
            summary_b["latency_stats"]["p50"],
            fmt=".2f",
            suffix=" s",
        )
    )
    lines.append(
        row(
            "p95 Latency",
            summary_a["latency_stats"]["p95"],
            summary_b["latency_stats"]["p95"],
            fmt=".2f",
            suffix=" s",
        )
    )
    lines.append(
        row(
            "p99 Latency",
            summary_a["latency_stats"].get(
                "p99", summary_a["latency_stats"].get("max", 0)
            ),
            summary_b["latency_stats"].get(
                "p99", summary_b["latency_stats"].get("max", 0)
            ),
            fmt=".2f",
            suffix=" s",
        )
    )
    lines.append(
        row(
            "Mean Latency",
            summary_a["latency_stats"]["mean"],
            summary_b["latency_stats"]["mean"],
            fmt=".2f",
            suffix=" s",
        )
    )
    lines.append(
        row(
            "Total Tokens",
            summary_a["total_tokens"],
            summary_b["total_tokens"],
            fmt=",d",
        )
    )
    lines.append(
        row(
            "Total Cost",
            summary_a["total_cost"],
            summary_b["total_cost"],
            fmt=".6f",
            suffix=" $",
        )
    )
    lines.append(
        row(
            "Cost per 1k Q",
            summary_a["cost_per_1k"],
            summary_b["cost_per_1k"],
            fmt=".6f",
            suffix=" $",
        )
    )

    lines += [""]

    if summary_b.get("cache"):
        c = summary_b["cache"]
        lines += [
            "## Cache Performance",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Hit rate | {c['hit_rate']:.1%} ({c['hit_count']}/{summary_b['n']}) |",
            f"| False-positive rate | {c['false_positive_rate']:.1%} ({c['false_positive_count']} of {c['hit_count']} hits) |",
            f"| Cost saved | ${c['cost_saved']:.6f} |",
            f"| Hit p50 latency | {c['hit_latency_stats']['p50']:.2f} s |",
            f"| Miss p50 latency | {c['miss_latency_stats']['p50']:.2f} s |",
            "",
            "### Hit Rate by Relationship",
            "",
            "| Relationship | Count | Hits | Hit Rate |",
            "|--------------|-------|------|----------|",
        ]
        for rel in sorted(c["by_relationship"]):
            d = c["by_relationship"][rel]
            rate = d["hit_count"] / d["count"] if d["count"] else 0
            lines.append(f"| {rel} | {d['count']} | {d['hit_count']} | {rate:.1%} |")
        lines += [""]

    lines += [
        "## Per Query Type",
        "",
        f"| Query Type | {label_a} Score | {label_b} Score | Δ |",
        f"|------------|--------------|--------------|---|",
    ]
    all_qtypes = sorted(set(summary_a["qtype_scores"]) | set(summary_b["qtype_scores"]))
    for qt in all_qtypes:
        sa = summary_a["qtype_scores"].get(qt, {}).get("mean_score", 0)
        sb = summary_b["qtype_scores"].get(qt, {}).get("mean_score", 0)
        lines.append(f"| {qt} | {sa:.3f} | {sb:.3f} | {sb - sa:+.3f} |")

    lines += [""]

    md = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison report from two experiment JSONL files"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-discover latest two JSONL files in results/jsonl/",
    )
    parser.add_argument("jsonl_a", nargs="?", help="Path to first JSONL results file")
    parser.add_argument(
        "--label-a", default="RAG Baseline", help="Label for first experiment"
    )
    parser.add_argument("jsonl_b", nargs="?", help="Path to second JSONL results file")
    parser.add_argument(
        "--label-b", default="Semantic Cache", help="Label for second experiment"
    )
    parser.add_argument("--output-dir", default="docs", help="Output directory")
    parser.add_argument("--prefix", default="report", help="Output filename prefix")
    args = parser.parse_args()

    if args.auto:
        jsonl_dir = Path("results") / "jsonl"
        if not jsonl_dir.exists():
            print(
                "Error: results/jsonl/ directory not found. Run experiments first.",
                file=sys.stderr,
            )
            sys.exit(1)
        jsonl_files = sorted(jsonl_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if len(jsonl_files) < 2:
            print(
                "Error: need at least 2 JSONL files in results/jsonl/.", file=sys.stderr
            )
            sys.exit(1)
        args.jsonl_a = str(jsonl_files[-2])
        args.jsonl_b = str(jsonl_files[-1])
        if (
            "rag_baseline" in jsonl_files[-2].name.lower()
            and "semantic_cache" in jsonl_files[-1].name.lower()
        ):
            pass
        elif (
            "semantic_cache" in jsonl_files[-2].name.lower()
            and "rag_baseline" in jsonl_files[-1].name.lower()
        ):
            args.jsonl_a, args.jsonl_b = args.jsonl_b, args.jsonl_a

    if not args.jsonl_a or not args.jsonl_b:
        parser.error("provide two JSONL paths or use --auto")

    records_a = load_jsonl(args.jsonl_a)
    records_b = load_jsonl(args.jsonl_b)

    summary_a = compute_summary(records_a)
    summary_b = compute_summary(records_b)

    is_cache_a = bool(summary_a.get("cache"))
    is_cache_b = bool(summary_b.get("cache"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    # Auto-detect model from the most recent experiment report
    model = ""
    report_glob = sorted(
        Path("results/reports").glob("*_report.md"), key=lambda p: p.stat().st_mtime
    )
    if report_glob:
        model = _detect_model(report_glob[-1])

    html_path = generate_html(
        args.label_a,
        summary_a,
        is_cache_a,
        args.label_b,
        summary_b,
        is_cache_b,
        str(out_dir / f"{args.prefix}.html"),
        model=model,
    )
    md_path = generate_markdown(
        args.label_a,
        summary_a,
        args.label_b,
        summary_b,
        str(out_dir / f"{args.prefix}.md"),
    )

    print(f"HTML report: {html_path}")
    print(f"Markdown report: {md_path}")


if __name__ == "__main__":
    main()
