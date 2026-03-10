from transformers import pipeline

class SentinexAnalyzer:
    def __init__(self):
        print("Loading SENTINEX models...")
        self.emotion_pipe = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=None
        )
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        self.sarcasm_pipe = pipeline(
            "text-classification",
            model="jkhan447/sarcasm-detection-RoBerta-base-POS"
        )
        self.context_history = []

        self.PSYCH_MARKERS = {
            "hopelessness": ["hopeless","pointless","no point","give up","worthless","nothing matters","cant go on","no reason"],
            "isolation":    ["alone","nobody","no one cares","lonely","isolated","no friends","nobody understands"],
            "self_blame":   ["my fault","blame myself","i am stupid","i am useless","hate myself","all my fault"],
            "sleep":        ["cant sleep","cry myself to sleep","insomnia","tired all the time","exhausted","no energy","cry at night"],
            "pessimism":    ["never get better","always like this","no future","wont improve","why do i still","what is the point"],
            "dissociation": ["feel numb","disconnected","empty inside","dont feel anything","just going through"]
        }

        self.CRITICAL_PHRASES = [
            "cry myself to sleep","want to die","end it all","no reason to live",
            "better off dead","cant take it anymore","give up on life","feel nothing anymore",
            "completely empty","numb inside","disappear forever","nobody would miss me"
        ]

        self.SARCASM_PHRASES = [
            "best week ever","best day ever","just perfect","oh great","oh wonderful",
            "yeah right","totally fine","absolutely fantastic","couldnt be better",
            "best thing ever","so amazing","what a great"
        ]
        print("All models loaded!")

    def detect_psych_markers(self, text):
        text_lower = text.lower()
        found = {}
        for marker, keywords in self.PSYCH_MARKERS.items():
            hits = [kw for kw in keywords if kw in text_lower]
            if hits:
                found[marker] = hits
        return found

    def detect_critical_phrases(self, text):
        text_lower = text.lower()
        return [p for p in self.CRITICAL_PHRASES if p in text_lower]

    def detect_sarcasm_phrases(self, text):
        text_lower = text.lower()
        return any(p in text_lower for p in self.SARCASM_PHRASES)

    def compute_risk(self, emotions, sentiment, is_sarcasm, psych_markers, critical_phrases, context_history):
        risk_score = 0.0
        HIGH_RISK = ["sadness","fear","disgust","anger"]

        # Critical phrase override
        if critical_phrases:
            risk_score += len(critical_phrases) * 0.3

        # Sarcasm flip
        effective_sentiment = sentiment[0]["label"]
        if is_sarcasm and effective_sentiment == "positive":
            effective_sentiment = "negative"
            risk_score += 0.25

        # Emotions
        for em in emotions:
            if em["label"].lower() in HIGH_RISK:
                risk_score += em["score"] * 0.35

        # Sentiment
        if effective_sentiment == "negative":
            risk_score += sentiment[0]["score"] * 0.25

        # Psychological markers
        risk_score += len(psych_markers) * 0.12

        # Context trend
        if len(context_history) >= 3:
            neg_count = sum(1 for h in context_history[-5:] if h["risk"] > 0.4)
            if neg_count >= 3:
                risk_score += 0.2

        risk_score = min(risk_score, 1.0)

        if risk_score < 0.3:
            level  = "🟢 LOW"
            advice = "You seem to be doing well. Keep it up!"
        elif risk_score < 0.55:
            level  = "🟡 MODERATE"
            advice = "Some stress detected. Consider talking to someone you trust."
        elif risk_score < 0.75:
            level  = "🟠 HIGH"
            advice = "High distress detected. Please talk to a counselor or someone you trust."
        else:
            level  = "🔴 CRITICAL"
            advice = "Critical distress detected. Please reach out to a mental health professional immediately. You are not alone. iCall: 9152987821"

        return round(risk_score, 3), level, advice

    def analyze(self, text):
        if not text or len(text.strip()) < 5:
            return None

        emotions         = self.emotion_pipe(text)[0]
        sentiment        = self.sentiment_pipe(text)
        sarcasm_result   = self.sarcasm_pipe(text)[0]
        rule_sarcasm     = self.detect_sarcasm_phrases(text)
        is_sarcasm       = sarcasm_result["label"] == "SARCASM" or rule_sarcasm
        psych_markers    = self.detect_psych_markers(text)
        critical_phrases = self.detect_critical_phrases(text)

        risk_score, risk_level, advice = self.compute_risk(
            emotions, sentiment, is_sarcasm,
            psych_markers, critical_phrases, self.context_history
        )

        emotions_sorted = sorted(emotions, key=lambda x: x["score"], reverse=True)

        self.context_history.append({
            "text":    text[:100],
            "risk":    risk_score,
            "emotion": emotions_sorted[0]["label"],
            "sarcasm": is_sarcasm
        })
        if len(self.context_history) > 10:
            self.context_history.pop(0)

        return {
            "emotions":        emotions_sorted,
            "sentiment":       sentiment[0],
            "is_sarcasm":      is_sarcasm,
            "sarcasm_conf":    round(sarcasm_result["score"], 3),
            "psych_markers":   psych_markers,
            "critical_phrases":critical_phrases,
            "risk_score":      risk_score,
            "risk_level":      risk_level,
            "advice":          advice,
            "context_history": self.context_history
        }
