"""
components/urgency_classifier.py
────────────────────────────────
A single helper that feeds **full call context** to an Ollama LLM
(e.g. llama2:7b) and extracts three values:

    • category        (fire, accident, theft, …)
    • urgency_level   (critical, high, moderate, low, non_emergency)
    • confidence (%)  model‑reported certainty 0‑100

The call‑context you must pass in is a dictionary:

{
    "full_transcript": str,
    "speaker_segments": [
        { "speaker": str, "start": float, "end": float,
          "text": str, "emotion": str, "confidence": float }
    ],
    "background": [
        { "label": str, "confidence": float }
    ]
}

Dependencies  (add to requirements.txt if missing)
──────────────────────────────────────────────────
langchain-core
langchain-ollama   (pip install langchain-ollama)
ollama             (the local server must be running)
"""

from __future__ import annotations

import json
import re
from typing import Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# ──────────────────────────────────────────────────────────────
#  Prompt template
# ──────────────────────────────────────────────────────────────
_PROMPT_TMPL = """
You are an intelligent emergency‑call analyzer.

You receive a JSON object that contains:
• "full_transcript"  – entire call transcript
• "speaker_segments" – list of {{speaker,start,end,text,emotion,confidence}}
• "background"       – list of {{label,confidence}}

Analyse all these signals and return ONLY valid JSON (no markdown):

{{
  "category": "<fire | burglary | domestic_violence | cyber_fraud | medical_emergency | accident | theft | street_violence | kidnap | other>",
  "urgency_level": "<critical | high | moderate | low | non_emergency>",
  "confidence": <0‑100 float>
}}

---------------
CALL CONTEXT:
```json
{context}
```
"""

_prompt = ChatPromptTemplate.from_template(_PROMPT_TMPL.strip())

# Configure your local Ollama model name here
_llm = OllamaLLM(
    model="gemma3:4b", 
    temperature=0.1,
    num_predict=300,
    num_ctx=4096,
)

_chain = _prompt | _llm

# ──────────────────────────────────────────────────────────────
#  Public helper
# ──────────────────────────────────────────────────────────────
def classify_urgency(context_dict: dict) -> Tuple[str, str, float]:
    """
    Parameters
    ----------
    context_dict : dict
        {
            "full_transcript": str,
            "speaker_segments": [...],
            "background": [...]
        }
    
    Returns
    -------
    category        : str   e.g. "fire"
    urgency_level   : str   e.g. "critical"
    confidence      : float 0‑100 (clamped)
    """
    # 1) Call LLM
    raw = _chain.invoke(
        {"context": json.dumps(context_dict, ensure_ascii=False)}
    ).strip()
    
    # 2) Extract JSON subsequence
    json_str = None
    if raw.startswith("{"):
        json_str = raw
    else:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            json_str = m.group()
    
    if not json_str:
        # fallback
        return "other", "moderate", 50.0
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return "other", "moderate", 50.0
    
    # 3) Normalise / validate
    cat = str(data.get("category", "other")).lower().strip()
    lvl = str(data.get("urgency_level", "moderate")).lower().strip()
    conf = float(data.get("confidence", 50.0))
    
    valid_cat = {
        "fire", "burglary", "domestic_violence", "cyber_fraud",
        "medical_emergency", "accident", "theft",
        "street_violence", "other",
    }
    valid_lvl = {"critical", "high", "moderate", "low", "non_emergency"}
    
    if cat not in valid_cat:
        cat = "other"
    if lvl not in valid_lvl:
        lvl = "moderate"
    conf = max(0.0, min(conf, 100.0))   # clamp 0‑100
    
    return cat, lvl, conf

# ──────────────────────────────────────────────────────────────
#  Quick CLI test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo_context = {
        "full_transcript": "Emergency. Don't open the door. I have an emergency. There is an intruder in my house and you need to come right now. Your address?",
        "speaker_segments": [
            {"speaker": "SPEAKER_00", "start": 0.5, "end": 1.1,
             "text": "emergency", "emotion": "Sad", "confidence": 0.24},
            {"speaker": "SPEAKER_01", "start": 1.6, "end": 6.1,
             "text": "Don't open the door. I have an emergency. There is an intruder in my house and you need to come right now.", "emotion": "Fear", "confidence": 0.35},
            {"speaker": "SPEAKER_00", "start": 6.1, "end": 6.5,
             "text": "your address", "emotion": "Neutral", "confidence": 0.19}
        ],
        "background": [],
    }
    print("Testing urgency classification …")
    print(classify_urgency(demo_context))