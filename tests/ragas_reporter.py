"""
RAGAS HTML Report Generator

Converts ragas_results.json into an HTML dashboard showing:
- Metric scores
- Pass/fail status
- Threshold lines
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_PATH = PROJECT_ROOT / "tests" / "ragas_results.json"
REPORT_PATH = PROJECT_ROOT / "tests" / "ragas_report.html"


def generate_report():
    """Generate HTML report from RAGAS results."""
    
    # Load results
    if not RESULTS_PATH.exists():
        print(f"Error: Results file not found: {RESULTS_PATH}")
        print("Run ragas_evaluation.py first.")
        sys.exit(1)
    
    with open(RESULTS_PATH, 'r') as f:
        data = json.load(f)
    
    scores = data["scores"]
    thresholds = data["thresholds"]
    all_passed = data["all_passed"]
    timestamp = data["timestamp"]
    num_samples = data["num_samples"]
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGAS Evaluation Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 2rem;
            color: #e0e0e0;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            margin-bottom: 2rem;
        }}
        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(135deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .header .meta {{
            color: #888;
            font-size: 0.9rem;
        }}
        .status-badge {{
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 2rem;
            font-weight: bold;
            font-size: 1.2rem;
            margin: 1rem 0;
        }}
        .status-pass {{
            background: linear-gradient(135deg, #00c853, #00e676);
            color: #000;
        }}
        .status-fail {{
            background: linear-gradient(135deg, #ff1744, #ff5252);
            color: #fff;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 1rem;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }}
        .metric-card h3 {{
            font-size: 0.9rem;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }}
        .metric-score {{
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }}
        .metric-pass {{
            color: #00e676;
        }}
        .metric-fail {{
            color: #ff5252;
        }}
        .threshold-bar {{
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            margin-top: 1rem;
            position: relative;
            overflow: hidden;
        }}
        .threshold-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        .threshold-fill.pass {{
            background: linear-gradient(90deg, #00c853, #00e676);
        }}
        .threshold-fill.fail {{
            background: linear-gradient(90deg, #ff1744, #ff5252);
        }}
        .threshold-marker {{
            position: absolute;
            top: -4px;
            width: 2px;
            height: 16px;
            background: #fff;
        }}
        .threshold-label {{
            font-size: 0.75rem;
            color: #888;
            margin-top: 0.5rem;
            display: flex;
            justify-content: space-between;
        }}
        .samples-section {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-top: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .samples-section h2 {{
            margin-bottom: 1rem;
            color: #00d2ff;
        }}
        .sample-count {{
            font-size: 0.9rem;
            color: #888;
        }}
        footer {{
            text-align: center;
            margin-top: 2rem;
            color: #666;
            font-size: 0.8rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 RAGAS Evaluation Report</h1>
            <p class="meta">Generated: {timestamp}</p>
            <p class="meta">Samples: {num_samples}</p>
            <div class="status-badge {'status-pass' if all_passed else 'status-fail'}">
                {'✅ ALL METRICS PASSED' if all_passed else '❌ EVALUATION FAILED'}
            </div>
        </div>
        
        <div class="metrics-grid">
"""
    
    for metric, score in scores.items():
        threshold = thresholds[metric]
        passed = score >= threshold
        status_class = "pass" if passed else "fail"
        fill_width = min(score * 100, 100)
        threshold_pos = threshold * 100
        
        html += f"""
            <div class="metric-card">
                <h3>{metric.replace('_', ' ')}</h3>
                <div class="metric-score metric-{status_class}">{score:.2%}</div>
                <div class="threshold-bar">
                    <div class="threshold-fill {status_class}" style="width: {fill_width}%"></div>
                    <div class="threshold-marker" style="left: {threshold_pos}%"></div>
                </div>
                <div class="threshold-label">
                    <span>0%</span>
                    <span>Threshold: {threshold:.0%}</span>
                    <span>100%</span>
                </div>
            </div>
"""
    
    html += f"""
        </div>
        
        <div class="samples-section">
            <h2>Evaluation Summary</h2>
            <p class="sample-count">Evaluated {num_samples} samples across BNM policy documents.</p>
            <table style="width: 100%; margin-top: 1rem; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <th style="text-align: left; padding: 0.5rem 0; color: #888;">Metric</th>
                    <th style="text-align: right; padding: 0.5rem 0; color: #888;">Score</th>
                    <th style="text-align: right; padding: 0.5rem 0; color: #888;">Threshold</th>
                    <th style="text-align: right; padding: 0.5rem 0; color: #888;">Status</th>
                </tr>
"""
    
    for metric, score in scores.items():
        threshold = thresholds[metric]
        passed = score >= threshold
        status_color = "#00e676" if passed else "#ff5252"
        status_text = "PASS" if passed else "FAIL"
        
        html += f"""
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 0.75rem 0;">{metric}</td>
                    <td style="text-align: right; padding: 0.75rem 0;">{score:.4f}</td>
                    <td style="text-align: right; padding: 0.75rem 0;">{threshold:.2f}</td>
                    <td style="text-align: right; padding: 0.75rem 0; color: {status_color}; font-weight: bold;">{status_text}</td>
                </tr>
"""
    
    html += """
            </table>
        </div>
        
        <footer>
            Generated by RAGAS Evaluation Pipeline | Bank Negara Malaysia RAG System
        </footer>
    </div>
</body>
</html>
"""
    
    # Write HTML file
    with open(REPORT_PATH, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
