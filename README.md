
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:87CEEB,50:B0E0E6,100:FFFDD0&height=220&section=header&text=SENTINEX&fontSize=85&fontColor=1a1a2e&fontAlignY=35&desc=Mental%20Health%20Intelligence%20AI%20v2.0&descAlignY=60&descSize=20&animation=fadeIn" width="100%"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Orbitron&weight=900&size=24&duration=3000&pause=800&color=1E90FF&center=true&vCenter=true&multiline=true&width=750&height=120&lines=🧠+Emotion+Detection+%7C+Risk+Scoring;🎭+Sarcasm+Intelligence+%7C+NLP+AI;🔴+CRITICAL+to+🟢+LOW+—+Real+Time;Built+with+3+Transformer+Models)](https://git.io/typing-svg)

<br/>

![Python](https://img.shields.io/badge/Python-3.11-87CEEB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFFACD?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-UI-87CEEB?style=for-the-badge&logo=gradio&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-B0E0E6?style=for-the-badge)

<br/>

> **🧠 The AI that reads between the lines — detecting hidden pain, masked depression, and silent cries for help.**

<br/>

[![Live Demo](https://img.shields.io/badge/🚀_LIVE_DEMO-HuggingFace_Space-FFD700?style=for-the-badge)](https://huggingface.co/spaces/siddhantchandorkar/sentinex-mental-health-ai)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/siddhantchandorkar752-ai/sentinex-mental-health-ai)

</div>

---

## 🔥 WHAT IS SENTINEX?

```
╔══════════════════════════════════════════════════════════════════════╗
║     SENTINEX — Mental Health Intelligence System v2.0               ║
║     "Not just what you say — but what you MEAN."                    ║
║     Built with 3 Transformer Models + Rule-Based Psychology         ║
║     Detects: Emotion • Sarcasm • Risk • Hidden Distress             ║
╚══════════════════════════════════════════════════════════════════════╝
```

SENTINEX is an advanced **NLP-powered Mental Health Analysis AI** that goes beyond surface-level sentiment. It understands **sarcasm**, detects **psychological markers**, tracks **session mood trends**, and delivers a precise **4-tier risk assessment** — from 🟢 LOW to 🔴 CRITICAL.

> Most AI tools see *what you say*. SENTINEX sees *what you mean.*

---

## 😔 PROBLEM STATEMENT

Most sentiment analysis tools fail at detecting **masked depression** — when someone says *"I'm fine"* but isn't. Traditional NLP misses:

- Sarcasm hiding real pain — *"Best week ever!"* after losing a job
- Hidden psychological distress markers — *"cry myself to sleep"*
- Gradual mood deterioration across a conversation
- The gap between expressed positivity and underlying suffering

**SENTINEX solves this** using a multi-model fusion approach with rule-based psychology layers.

---

## ⚡ CORE FEATURES

| Feature | Description |
|---------|-------------|
| 🎭 **Sarcasm Engine** | Detects masked negativity — *"Best week ever"* → SARCASM DETECTED |
| 😢 **7-Class Emotion** | Sadness, Joy, Fear, Anger, Disgust, Surprise, Neutral |
| 🧬 **Psych Markers** | Hopelessness, Isolation, Self-Blame, Sleep Issues, Pessimism, Dissociation |
| 📈 **Context Memory** | Tracks last 10 messages — escalating sadness triggers automatic risk boost |
| 🚨 **4-Tier Risk System** | LOW → MODERATE → HIGH → CRITICAL with crisis helpline resources |
| 💬 **Session History** | Visual mood timeline with trend analysis across entire conversation |
| 🔍 **Critical Phrase Detection** | Flags phrases like "cry myself to sleep", "no reason to live" instantly |

---

## 🚨 RISK LEVEL SYSTEM

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  🟢  LOW      Score: 0.00–0.30  "You seem to be doing well!"   │
│  🟡  MODERATE Score: 0.30–0.55  "Some stress detected."        │
│  🟠  HIGH     Score: 0.55–0.75  "Please seek support."         │
│  🔴  CRITICAL Score: 0.75–1.00  "Immediate help needed."       │
│                        iCall India: 9152987821                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 ARCHITECTURE

```
INPUT TEXT
    │
    ▼
┌──────────────────────────────────────────────────────┐
│                  SENTINEX PIPELINE                   │
│                                                      │
│   ┌──────────────┐        ┌───────────────┐         │
│   │   Emotion    │        │   Sentiment   │         │
│   │DistilRoBERTa │        │ Twitter-RoBERTa│         │
│   │  7 classes   │        │  POS/NEG/NEU  │         │
│   └──────┬───────┘        └──────┬────────┘         │
│          │                       │                  │
│   ┌──────▼───────┐        ┌──────▼────────┐         │
│   │   Sarcasm    │        │  Psych Marker │         │
│   │ RoBERTa-POS  │        │    Scanner    │         │
│   │ + Rule Engine│        │  6 categories │         │
│   └──────┬───────┘        └──────┬────────┘         │
│          │                       │                  │
│          └──────────┬────────────┘                  │
│                     ▼                               │
│          ┌──────────────────┐                       │
│          │   Risk Scorer    │                       │
│          │  Context Memory  │ ← Last 10 messages    │
│          │  Critical Phrases│                       │
│          └──────────┬───────┘                       │
│                     ▼                               │
│   🟢 LOW / 🟡 MODERATE / 🟠 HIGH / 🔴 CRITICAL      │
└──────────────────────────────────────────────────────┘
```

---

## 🤖 MODELS USED

| Model | Purpose | Parameters |
|-------|---------|------------|
| `j-hartmann/emotion-english-distilroberta-base` | 7-class Emotion Detection | 82M |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment Analysis | 125M |
| `jkhan447/sarcasm-detection-RoBerta-base-POS` | Sarcasm Detection | 125M |

> All models downloaded automatically from HuggingFace Hub on first run (~1.5GB total).

---

## 📁 PROJECT STRUCTURE

```
sentinex/
├── app.py                  ← Gradio web interface
├── models/
│   ├── __init__.py
│   └── analyzer.py         ← Core SENTINEX engine (risk scorer, psych markers, sarcasm)
├── requirements.txt        ← Python dependencies
├── Dockerfile              ← HuggingFace Docker deployment
└── README.md               ← You are here
```

---

## 🛠️ INSTALLATION

### Prerequisites
- Python 3.9+
- pip
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/siddhantchandorkar752-ai/sentinex-mental-health-ai.git
cd sentinex-mental-health-ai
```

### Step 2 — Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install transformers torch gradio scipy
```

> ⚠️ First run downloads ~1.5GB of transformer models automatically. Subsequent runs use cache.

---

## 🚀 USAGE

### Run Locally
```bash
python app.py
```
Open browser at: **http://127.0.0.1:7860**

### Run with Docker
```bash
docker build -t sentinex .
docker run -p 7860:7860 sentinex
```

### Use Online — No Installation Required
[![Live Demo](https://img.shields.io/badge/🚀_Try_SENTINEX_Live-HuggingFace-FFD700?style=for-the-badge)](https://huggingface.co/spaces/siddhantchandorkar/sentinex-mental-health-ai)

---

## 🧪 EXAMPLE OUTPUT

```
Input:
  "I finally got everything I wanted. Nice house, good job, great friends.
   So why do I still cry myself to sleep every night?"

─────────────────────────────────────────────
SENTINEX ANALYSIS REPORT
─────────────────────────────────────────────
Emotion:        NEUTRAL 45% | JOY 29% | SADNESS 6%
Sentiment:      POSITIVE (0.91) ← surface level
Sarcasm:        Not detected

Psychological Markers Detected:
  • SLEEP     → "cry myself to sleep"
  • PESSIMISM → "why do i still"

Risk Score:     0.567 / 1.0
Risk Level:     🟠 HIGH
Advice:         High distress detected. Please talk to a counselor.
─────────────────────────────────────────────
```

> Notice: Surface sentiment says POSITIVE — but SENTINEX correctly flags HIGH risk via psych markers. This is SENTINEX's core strength.

---

## 📊 TEST RESULTS

| Input | Expected | SENTINEX Output | Result |
|-------|----------|-----------------|--------|
| *"Best week ever! Lost job, cat died..."* | HIGH + SARCASM | 🟠 HIGH + SARCASM DETECTED | ✅ PASS |
| *"Got everything... why do I cry myself to sleep?"* | HIGH | 🟠 HIGH + SLEEP + PESSIMISM markers | ✅ PASS |
| *"Got my dream job! Years of hard work paid off!"* | LOW | 🟢 LOW + JOY 90% | ✅ PASS |
| *"Nobody understands. It's all my fault. Can't sleep."* | CRITICAL | 🔴 CRITICAL + 3 psych markers | ✅ PASS |
| *"Went to market, cooked dinner, watched TV."* | LOW | 🟢 LOW + NEUTRAL | ✅ PASS |

**Overall Accuracy: 5/5 test cases passed ✅**

---

## 🔮 FUTURE IMPROVEMENTS

- [ ] Voice input — audio → text → SENTINEX pipeline
- [ ] Multilingual support (Hindi, Spanish, French)
- [ ] Fine-tuned custom model on clinical mental health datasets
- [ ] Long-term user trend tracking with database storage
- [ ] Therapist dashboard with exportable session reports
- [ ] Mobile app integration (React Native)
- [ ] Real-time chat monitoring API

---

## 🛠️ TECH STACK

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-87CEEB?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-RoBERTa-4682B4?style=for-the-badge)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)

---

## 🤝 CONTRIBUTING

Contributions are welcome! Here's how:

```bash
# 1. Fork the repository
# 2. Create your feature branch
git checkout -b feature/AmazingFeature

# 3. Commit your changes
git commit -m 'Add AmazingFeature'

# 4. Push to the branch
git push origin feature/AmazingFeature

# 5. Open a Pull Request on GitHub
```

---

## 📄 LICENSE

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 👨‍💻 AUTHOR

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:87CEEB,100:FFFDD0&height=60&text=Siddhant%20Chandorkar&fontSize=28&fontColor=1a1a2e&fontAlign=50&fontAlignY=50" width="500"/>

<br/><br/>

[![GitHub](https://img.shields.io/badge/GitHub-siddhantchandorkar752--ai-181717?style=for-the-badge&logo=github)](https://github.com/siddhantchandorkar752-ai)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-siddhantchandorkar-FFFACD?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/siddhantchandorkar)

<br/>

*"I don't just build AI. I build AI that understands humans."*

</div>

---

<div align="center">

> ⚠️ **Disclaimer:** SENTINEX is an AI research tool and is **NOT** a substitute for professional mental health care. If you or someone you know is in crisis, please contact a mental health professional immediately.
>
> 🇮🇳 **iCall India:** 9152987821 &nbsp;|&nbsp; **Vandrevala Foundation:** 1860-2662-345 &nbsp;|&nbsp; **iCall:** 9152987821

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:FFFDD0,50:B0E0E6,100:87CEEB&height=120&section=footer&text=SENTINEX%20v2.0&fontSize=30&fontColor=1a1a2e&fontAlignY=65&animation=fadeIn" width="100%"/>

</div>
