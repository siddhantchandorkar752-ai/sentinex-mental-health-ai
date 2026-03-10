

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:87CEEB,50:B0E0E6,100:FFFDD0&height=200&section=header&text=SENTINEX&fontSize=80&fontColor=1a1a2e&fontAlignY=35&desc=Mental%20Health%20Intelligence%20AI&descAlignY=60&descSize=22&animation=fadeIn" width="100%"/>

<br/>

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Orbitron&weight=900&size=28&duration=3000&pause=800&color=1E90FF&center=true&vCenter=true&multiline=true&width=700&height=100&lines=🧠+Emotion+Detection+%7C+Risk+Scoring;🎭+Sarcasm+Intelligence+%7C+NLP+AI;🔴+CRITICAL+to+🟢+LOW+—+Real+Time)](https://git.io/typing-svg)

<br/>

![Python](https://img.shields.io/badge/Python-3.11-87CEEB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Transformers-FFFDD0?style=for-the-badge&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-UI-87CEEB?style=for-the-badge&logo=gradio&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-B0E0E6?style=for-the-badge)

<br/>

> **🧠 The AI that reads between the lines — detecting hidden pain, masked depression, and silent cries for help.**

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

---

## 😔 PROBLEM STATEMENT

Most sentiment analysis tools fail at detecting **masked depression** — when someone says *"I'm fine"* but isn't. Traditional NLP misses:

- Sarcasm hiding real pain (*"Best week ever!"* after losing a job)
- Hidden psychological distress markers (*"cry myself to sleep"*)
- Gradual mood deterioration over a conversation
- The gap between expressed positivity and underlying suffering

**SENTINEX solves this** using a multi-model fusion approach with rule-based psychology.

---

## ⚡ CORE FEATURES

| Feature | Description |
|---------|-------------|
| 🎭 **Sarcasm Engine** | Detects masked negativity — *"Best week ever"* → SARCASM |
| 😢 **7-Class Emotion** | Sadness, Joy, Fear, Anger, Disgust, Surprise, Neutral |
| 🧬 **Psych Markers** | Hopelessness, Isolation, Self-Blame, Sleep Issues, Pessimism |
| 📈 **Context Memory** | Tracks last 10 messages — escalating sadness = risk boost |
| 🚨 **4-Tier Risk** | LOW → MODERATE → HIGH → CRITICAL with crisis resources |
| 💬 **Session History** | Visual mood timeline across the conversation |

---

## 🚨 RISK LEVEL SYSTEM

```
🟢  LOW      Score: 0.00 – 0.30   "You seem to be doing well. Keep it up!"
🟡  MODERATE Score: 0.30 – 0.55   "Some stress detected. Talk to someone."
🟠  HIGH     Score: 0.55 – 0.75   "High distress. Please seek support."
🔴  CRITICAL Score: 0.75 – 1.00   "Immediate help needed. iCall: 9152987821"
```

---

## 🧠 ARCHITECTURE

```
INPUT TEXT
    │
    ▼
┌─────────────────────────────────────────────────┐
│              SENTINEX PIPELINE                  │
│  ┌─────────────┐  ┌──────────────┐             │
│  │  Emotion    │  │  Sentiment   │             │
│  │DistilRoBERTa│  │  RoBERTa     │             │
│  │  7 classes  │  │  (Twitter)   │             │
│  └──────┬──────┘  └──────┬───────┘             │
│  ┌──────▼──────┐  ┌──────▼───────┐             │
│  │  Sarcasm    │  │   Psych      │             │
│  │  RoBERTa +  │  │   Marker     │             │
│  │  Rule Engine│  │   Scanner    │             │
│  └──────┬──────┘  └──────┬───────┘             │
│         └────────┬────────┘                    │
│                  ▼                             │
│         ┌────────────────┐                    │
│         │  Risk Scorer   │                    │
│         │ Context Memory │                    │
│         └────────┬───────┘                    │
│    🟢 LOW / 🟡 MODERATE / 🟠 HIGH / 🔴 CRITICAL│
└─────────────────────────────────────────────────┘
```

---

## 🤖 MODELS USED

| Model | Purpose | Parameters |
|-------|---------|------------|
| `j-hartmann/emotion-english-distilroberta-base` | 7-class Emotion Detection | 82M |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Sentiment Analysis | 125M |
| `jkhan447/sarcasm-detection-RoBerta-base-POS` | Sarcasm Detection | 125M |

---

## 📁 PROJECT STRUCTURE

```
sentinex/
├── app.py                  ← Gradio web interface
├── models/
│   ├── __init__.py
│   └── analyzer.py         ← Core SENTINEX engine
├── requirements.txt        ← Dependencies
├── Dockerfile              ← HuggingFace deployment
└── README.md
```

---

## 🛠️ INSTALLATION

### Prerequisites
- Python 3.9+
- pip
- Git

### Clone the Repository
```bash
git clone https://github.com/siddhantchandorkar752-ai/sentinex-mental-health-ai.git
cd sentinex-mental-health-ai
```

### Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install transformers torch gradio scipy
```

> ⚠️ First run will automatically download ~1.5GB of models from HuggingFace.

---

## 🚀 USAGE

### Run Locally
```bash
python app.py
```
Then open: **http://127.0.0.1:7860**

### Run with Docker
```bash
docker build -t sentinex .
docker run -p 7860:7860 sentinex
```

### Use Online (No Install Required)
🔗 **https://huggingface.co/spaces/siddhantchandorkar/sentinex-mental-health-ai**

---

## 🧪 EXAMPLE OUTPUT

```
Input:  "I finally got everything I wanted. Nice house, good job.
         So why do I still cry myself to sleep every night?"

Output:
  Emotion:       NEUTRAL 45% | JOY 29%
  Sentiment:     POSITIVE (0.91) — surface level positive
  Sarcasm:       Not detected
  Psych Markers:
    • SLEEP:     "cry myself to sleep"
    • PESSIMISM: "why do i still"
  Risk Score:    0.567 / 1.0
  Risk Level:    🟠 HIGH
  Advice:        High distress detected. Please talk to a counselor.
```

---

## 📊 TEST RESULTS

| Input | Expected | Output | Status |
|-------|----------|--------|--------|
| *"Best week ever! Lost job, cat died..."* | HIGH + SARCASM | 🟠 HIGH \| SARCASM ✅ | ✅ PASS |
| *"Got everything... why do I cry myself to sleep?"* | HIGH | 🟠 HIGH \| SLEEP + PESSIMISM | ✅ PASS |
| *"Got my dream job! So grateful!"* | LOW | 🟢 LOW \| JOY 90% | ✅ PASS |
| *"Nobody understands. It's all my fault. Can't sleep."* | CRITICAL | 🔴 CRITICAL \| 3 markers | ✅ PASS |
| *"Went to market, cooked dinner, watched TV."* | LOW | 🟢 LOW \| NEUTRAL | ✅ PASS |

**Accuracy: 5/5 test cases passed ✅**

---

## 🔮 FUTURE IMPROVEMENTS

- [ ] Voice input analysis (audio → text → SENTINEX)
- [ ] Multilingual support (Hindi, Spanish, French)
- [ ] Fine-tuned custom model on mental health datasets
- [ ] Long-term user trend tracking with database
- [ ] Therapist dashboard with session reports
- [ ] Mobile app integration

---

## 🛠️ TECH STACK

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-87CEEB?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-RoBERTa-87CEEB?style=for-the-badge)

---

## 🤝 CONTRIBUTING

Contributions are welcome!

```bash
git checkout -b feature/AmazingFeature
git commit -m 'Add AmazingFeature'
git push origin feature/AmazingFeature
# Then open a Pull Request
```

---

## 📄 LICENSE

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 AUTHOR

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:87CEEB,100:FFFDD0&height=60&text=Siddhant%20Chandorkar&fontSize=28&fontColor=1a1a2e&fontAlign=50&fontAlignY=50" width="500"/>

<br/><br/>

[![GitHub](https://img.shields.io/badge/GitHub-siddhantchandorkar752--ai-181717?style=for-the-badge&logo=github)](https://github.com/siddhantchandorkar752-ai)
[![HuggingFace](https://img.shields.io/badge/🤗-siddhantchandorkar-FFFDD0?style=for-the-badge&logoColor=black)](https://huggingface.co/siddhantchandorkar)

<br/>

*"I don't just build AI. I build AI that understands humans."*

</div>

---

> ⚠️ **Disclaimer:** SENTINEX is an AI research tool and is NOT a substitute for professional mental health care. If you or someone you know is in crisis, please contact a mental health professional immediately.
>
> 🇮🇳 **iCall India:** 9152987821 | **Vandrevala Foundation:** 1860-2662-345

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:FFFDD0,50:B0E0E6,100:87CEEB&height=120&section=footer&text=SENTINEX%20v2.0&fontSize=30&fontColor=1a1a2e&fontAlignY=65&animation=fadeIn" width="100%"/>
</div>
