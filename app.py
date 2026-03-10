import gradio as gr
import numpy as np
from models.analyzer import SentinexAnalyzer

analyzer = SentinexAnalyzer()

def analyze_text(text):
    if not text or len(text.strip()) < 5:
        return "Please enter at least a sentence.", "", "", ""

    result = analyzer.analyze(text)

    # Emotion bars
    emotion_html = "<div style='font-family:monospace;padding:10px'>"
    for em in result["emotions"]:
        pct   = int(em["score"] * 100)
        label = em["label"].upper()
        color = {
            "joy":      "#00ff9f",
            "sadness":  "#4488ff",
            "anger":    "#ff4444",
            "fear":     "#ff8800",
            "surprise": "#ffff00",
            "disgust":  "#aa44ff",
            "neutral":  "#888888"
        }.get(em["label"].lower(), "#888888")
        emotion_html += f"""
        <div style='margin:6px 0'>
            <span style='color:{color};width:110px;display:inline-block;font-weight:bold'>{label}</span>
            <div style='display:inline-block;background:{color};width:{pct*3}px;height:16px;border-radius:4px;vertical-align:middle'></div>
            <span style='color:#ccc;margin-left:8px'>{pct}%</span>
        </div>"""
    emotion_html += "</div>"

    # Sarcasm + psych markers
    sarcasm_color = "#ff4444" if result["is_sarcasm"] else "#00ff9f"
    sarcasm_text  = f"⚠️ SARCASM DETECTED ({result['sarcasm_conf']:.0%})" if result["is_sarcasm"] else f"✅ No Sarcasm ({result['sarcasm_conf']:.0%})"

    psych_html = ""
    if result["psych_markers"]:
        psych_html = "<div style='margin-top:10px;padding:8px;background:#1a1a1a;border-radius:8px'>"
        psych_html += "<p style='color:#ff8800;margin:0 0 6px 0'>⚠️ Psychological Markers Detected:</p>"
        for marker, keywords in result["psych_markers"].items():
            psych_html += f"<p style='color:#ffaa44;margin:2px 0'>• {marker.upper()}: {', '.join(keywords)}</p>"
        psych_html += "</div>"

    # Risk panel
    risk_color = {
        "🟢 LOW":      "#00ff9f",
        "🟡 MODERATE": "#ffaa00",
        "🟠 HIGH":     "#ff6600",
        "🔴 CRITICAL": "#ff0000"
    }.get(result["risk_level"], "#888")

    risk_html = f"""
    <div style='background:#111;padding:20px;border-radius:12px;border:2px solid {risk_color}'>
        <h2 style='color:{risk_color};margin:0'>Risk Level: {result["risk_level"]}</h2>
        <h3 style='color:#ccc'>Score: {result["risk_score"]:.3f} / 1.0</h3>
        <p style='color:#aaa'>Sentiment: <b style='color:{risk_color}'>{result["sentiment"]["label"].upper()}</b> ({result["sentiment"]["score"]:.2f})</p>
        <p style='color:{sarcasm_color}'>{sarcasm_text}</p>
        {psych_html}
        <div style='background:#222;padding:12px;border-radius:8px;margin-top:10px'>
            <p style='color:#fff;margin:0'>💡 {result["advice"]}</p>
        </div>
    </div>"""

    # Mood history
    history_html = "<div style='font-family:monospace;padding:10px'>"
    history_html += "<p style='color:#00d9ff;margin:0 0 8px 0'>📈 Session Mood History:</p>"
    if result["context_history"]:
        for i, h in enumerate(reversed(result["context_history"][-5:])):
            r = h["risk"]
            c = "#00ff9f" if r < 0.3 else "#ffaa00" if r < 0.55 else "#ff4444"
            bar = "█" * int(r * 20)
            history_html += f"<div style='color:{c};margin:3px 0'>[{len(result['context_history'])-i}] {bar} {r:.2f} — {h['emotion'].upper()}</div>"
    else:
        history_html += "<p style='color:#555'>No history yet</p>"
    history_html += "</div>"

    report = f"""SENTINEX v2 — Mental Health Analysis
{'='*45}
Risk Level    : {result["risk_level"]}
Risk Score    : {result["risk_score"]:.3f} / 1.0
Sentiment     : {result["sentiment"]["label"].upper()} ({result["sentiment"]["score"]:.2f})
Sarcasm       : {"YES ⚠️" if result["is_sarcasm"] else "NO ✅"}

Top Emotions:
{chr(10).join([f"  {e['label'].upper():<12} {int(e['score']*100)}%" for e in result["emotions"][:4]])}

Psychological Markers:
{chr(10).join([f"  • {m.upper()}: {', '.join(kw)}" for m,kw in result["psych_markers"].items()]) if result["psych_markers"] else "  None detected"}

Advice: {result["advice"]}
{'='*45}
⚠️ Not a medical diagnosis. Consult a professional."""

    return emotion_html, risk_html, history_html, report

with gr.Blocks(title="SENTINEX — Mental Health AI") as demo:
    gr.HTML("""
    <div style='text-align:center;padding:24px;background:linear-gradient(135deg,#0f0f1a,#1a1a2e);border-radius:14px;margin-bottom:20px'>
        <h1 style='color:#00ff9f;font-size:2.8em;margin:0'>🧠 SENTINEX v2</h1>
        <p style='color:#888;font-size:1.1em'>Mental Health Intelligence — Emotion + Sarcasm + Risk + Context Memory</p>
        <p style='color:#555'>Built by Siddhant Chandorkar</p>
    </div>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            text_input = gr.Textbox(
                label="Enter your thoughts, diary entry, or any text",
                placeholder="e.g. I have been feeling really low lately...",
                lines=6
            )
            analyze_btn = gr.Button("🔍 Analyze Mental State", variant="primary")
            gr.Examples(
                examples=[
                    ["Wow today was fantastic. I failed my exam, lost my wallet and laptop crashed. Couldn't have asked for a better day."],
                    ["I feel so hopeless and empty. I don't see the point of anything anymore."],
                    ["Today was amazing! I got the job offer and my family is so proud of me!"],
                    ["I am really stressed about exams. I can't sleep and keep overthinking everything."],
                    ["Nobody cares about me. I feel completely alone and it's all my fault."]
                ],
                inputs=text_input
            )
        with gr.Column(scale=1):
            emotion_output  = gr.HTML(label="Emotion Analysis")
            risk_output     = gr.HTML(label="Risk Assessment")
            history_output  = gr.HTML(label="Mood History")

    report_output = gr.Textbox(label="Full Report", lines=12)

    analyze_btn.click(
        analyze_text,
        inputs=text_input,
        outputs=[emotion_output, risk_output, history_output, report_output]
    )
    gr.HTML("<div style='text-align:center;color:#444;padding:10px'>⚠️ SENTINEX is not a substitute for professional mental health advice. If you are in crisis, please call a helpline.</div>")

demo.launch(server_name="0.0.0.0", server_port=7860)
