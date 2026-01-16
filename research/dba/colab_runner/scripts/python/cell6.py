# Send Notification
import json
import urllib.request
from pathlib import Path

results_file = output_dir / "results.json"
summary = ""
if results_file.exists():
    with open(results_file) as f:
        data = json.load(f)

    summary = "DBA Benchmark Results\n\n"
    for model_id in data.get('model_ids', []):
        s = data['summaries'].get(model_id, {})
        summary += f"{model_id}\n"
        summary += f"  Content Match: {s.get('content_match_rate', 0)*100:.1f}%\n"
        summary += f"  Avg Score: {s.get('soft_score_avg', 0):.2f}\n\n"

if NOTIFY_WEBHOOK:
    try:
        payload = {"text": f"Benchmark complete!\n\n{summary}\nResults: {output_dir}"}
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(NOTIFY_WEBHOOK, data=data, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req)
        print("Webhook notification sent!")
    except Exception as e:
        print(f"Failed to send webhook: {e}")

print(summary)
