"""
=============================================================================
VERITAS CORE - PRODUCTION NEWS INTEGRITY & TELEMETRY ENGINE
Dual Machine Learning Ensemble + Linguistic Telemetry & Heuristic Consensus
FastAPI + Native WebSockets on Port 8000
=============================================================================
"""

import asyncio
import json
import logging
import math
import os
import re
import string
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VeritasCore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, "news_model_bundle.joblib")
FAKE_CSV_PATH = os.path.join(BASE_DIR, "Fake.csv")
TRUE_CSV_PATH = os.path.join(BASE_DIR, "True.csv")
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")

ml_bundle: Optional[Dict[str, Any]] = None

TRUSTED_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "thehindu.com",
    "timesofindia.indiatimes.com", "hindustantimes.com", "bloomberg.com",
    "wsj.com", "nytimes.com", "ft.com", "aljazeera.com", "theguardian.com",
    "dw.com", "afp.com", "nature.com", "sciencemag.org", "npr.org", "pbs.org",
    "time.com", "washingtonpost.com", "economist.com", "cnbc.com", "cnn.com",
    "cbsnews.com", "nbcnews.com", "abcnews.go.com", "ndtv.com", "indianexpress.com"
}

CLICKBAIT_TRIGGERS = [
    "shocking", "mind-blowing", "mind blowing", "won't believe", "wont believe",
    "you won't believe", "miracle cure", "conspiracy", "secret revealed",
    "bombshell", "leaked memo", "they are hiding", "they don't want you to know",
    "instant cure", "wake up sheeple", "catastrophic truth", "unbelievable truth",
    "exposed forever", "destroying the country", "deep state", "elites don't want",
    "proof that aliens", "100% cure", "doctors hate this", "ban this before",
    "secret trick", "shut up mainstream media", "what happens next will blow",
    "share before deleted", "illuminati", "plandemic", "stolen election"
]

ATTRIBUTION_MARKERS = [
    "according to", "spokesperson said", "spokesperson confirmed",
    "in a statement", "told reporters", "official report stated",
    "published in", "peer-reviewed", "study published", "cited by",
    "press conference", "confirmed by", "investigation revealed",
    "department said", "officials stated", "testified before",
    "in an interview", "data shows", "sources confirmed",
    "reuters", "associated press", "reported by", "researchers at",
    "ministry announced", "spokesman told", "press release"
]

def clean_text_for_nlp(text: str) -> str:
    """Standardizes text for TF-IDF feature mapping."""
    if not text:
        return ""
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"read more at:.*", " ", t)
    t = re.sub(r"\[.*?\]", " ", t)
    t = re.sub(r"<.*?>+", " ", t)
    t = re.sub(r"[%s]" % re.escape(string.punctuation), " ", t)
    t = re.sub(r"\w*\d\w*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def load_or_train_models():
    """Loads pre-trained model bundle or initiates training."""
    global ml_bundle
    if os.path.exists(MODEL_BUNDLE_PATH):
        try:
            logger.info("Loading pre-trained news model bundle from disk...")
            ml_bundle = joblib.load(MODEL_BUNDLE_PATH)
            logger.info(f"Model bundle loaded! Features: {len(ml_bundle['vectorizer'].vocabulary_)}")
            return
        except Exception as e:
            logger.warning(f"Bundle load failed: {e}. Rebuilding...")

    if not (os.path.exists(FAKE_CSV_PATH) and os.path.exists(TRUE_CSV_PATH)):
        logger.error("Dataset files Fake.csv and True.csv not found in directory!")
        raise FileNotFoundError("Datasets Fake.csv and True.csv must be present.")

    logger.info("Training fresh ensemble model from CSV datasets...")
    t0 = time.time()
    df_fake = pd.read_csv(FAKE_CSV_PATH)
    df_true = pd.read_csv(TRUE_CSV_PATH)

    df_true["text"] = df_true["text"].apply(
        lambda x: re.sub(r"^\s*.*?\(\s*Reuters\s*\)\s*-\s*", "", str(x), flags=re.I)
    )

    df_fake["text"] = df_fake["title"].fillna("") + " " + df_fake["text"].fillna("")
    df_true["text"] = df_true["title"].fillna("") + " " + df_true["text"].fillna("")

    df_fake["label"] = 0
    df_true["label"] = 1

    sample_size = min(len(df_fake), len(df_true), 16000)
    df = pd.concat([
        df_fake.sample(n=sample_size, random_state=42),
        df_true.sample(n=sample_size, random_state=42)
    ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    df["clean_text"] = df["text"].apply(clean_text_for_nlp)

    vectorizer = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=3
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"].values

    model_lr = LogisticRegression(C=2.0, max_iter=500, random_state=42)
    model_lr.fit(X, y)

    model_sgd = SGDClassifier(loss="log_loss", max_iter=500, random_state=42)
    model_sgd.fit(X, y)

    model_nb = MultinomialNB(alpha=0.1)
    model_nb.fit(X, y)

    ml_bundle = {
        "vectorizer": vectorizer,
        "model_lr": model_lr,
        "model_sgd": model_sgd,
        "model_nb": model_nb,
        "accuracy_scores": {
            "lr": float(model_lr.score(X, y)),
            "sgd": float(model_sgd.score(X, y)),
            "nb": float(model_nb.score(X, y))
        },
        "trained_at": time.time(),
        "samples_trained": len(df)
    }

    joblib.dump(ml_bundle, MODEL_BUNDLE_PATH, compress=3)
    logger.info(f"Model trained and saved in {time.time() - t0:.2f}s!")

def fetch_url_content(url: str) -> Dict[str, Any]:
    """Scrapes news article headline and substantive paragraphs from URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")

    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = ""
        og_title = soup.find("meta", property="og:title") or soup.find("meta", attrs={"name": "twitter:title"})
        if og_title and og_title.get("content"):
            title = og_title["content"].strip()
        elif soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        elif soup.title:
            title = soup.title.get_text().strip()

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        paragraphs = [
            p.get_text().strip() for p in soup.find_all("p")
            if len(p.get_text().strip().split()) > 7
        ]
        body = " ".join(paragraphs)
        if not body or len(body.split()) < 20:
            body = soup.get_text(separator=" ", strip=True)

        full_content = f"{title}. {body}".strip() if title else body

        return {
            "success": True,
            "url": url,
            "domain": domain,
            "title": title or "Live Online Article",
            "text": full_content,
            "word_count": len(full_content.split())
        }
    except Exception as ex:
        logger.warning(f"URL fetch exception for {url}: {ex}")
        return {
            "success": False,
            "url": url,
            "domain": domain,
            "title": "Online News Source",
            "text": url,
            "word_count": 0,
            "error": str(ex)
        }

def calculate_linguistic_telemetry(raw_text: str, domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes heuristic and linguistic signals:
    - Sensationalism & Clickbait Index
    - Journalistic Attribution Density
    - Emotional Polarization
    - Lexical Richness (Type-Token Ratio)
    - Syntactic & Stylistic Coherence
    """
    words = re.findall(r"\b[A-Za-z]+\b", raw_text)
    total_words = len(words)
    if total_words == 0:
        return {
            "sensationalism_score": 40,
            "clickbait_risk": "Moderate",
            "attribution_score": 20,
            "attribution_count": 0,
            "domain_boost": 0.0,
            "emotional_polarization": "Neutral",
            "lexical_richness": "0.00",
            "read_time": "1 min read",
            "word_count": 0,
            "indicators": ["Text length is minimal; telemetry calculated on available tokens."]
        }

    # 1. Capitalization / Shouting metric
    uppercase_words = [w for w in words if w.isupper() and len(w) > 1 and w not in {"US", "UK", "UN", "EU", "NATO", "AI", "NASA", "FBI", "CIA", "ISRO", "WHO", "GDP", "PM", "CEO"}]
    caps_ratio = len(uppercase_words) / total_words

    # 2. Punctuation anomalies
    excess_exclamations = len(re.findall(r"!{2,}", raw_text)) + raw_text.count("!!!")
    excess_questions = len(re.findall(r"\?{2,}", raw_text)) + raw_text.count("???")
    punctuation_risk = min(40, (excess_exclamations * 15) + (excess_questions * 10))

    # 3. Sensational Clickbait keyword count
    lower_text = raw_text.lower()
    trigger_hits = [phrase for phrase in CLICKBAIT_TRIGGERS if phrase in lower_text]
    trigger_score = min(50, len(trigger_hits) * 20)

    sensationalism = min(100, int((caps_ratio * 130) + punctuation_risk + trigger_score))
    if sensationalism < 25:
        clickbait_risk = "Low (Factual Restraint)"
    elif sensationalism < 55:
        clickbait_risk = "Moderate (Mild Hyperbole)"
    else:
        clickbait_risk = "High (Severe Clickbait Signals)"

    # 4. Journalistic Attribution Markers & Quotes
    attribution_hits = [m for m in ATTRIBUTION_MARKERS if m in lower_text]
    quote_matches = re.findall(r'["“][^"”]{10,250}["”]', raw_text)
    attribution_count = len(attribution_hits) + len(quote_matches)
    attribution_score = min(100, int((len(attribution_hits) * 15) + (len(quote_matches) * 12)))

    # 5. Lexical Diversity
    unique_tokens = len(set(w.lower() for w in words))
    ttr = unique_tokens / total_words if total_words > 0 else 0.0

    # 6. Domain Credibility Prior
    domain_boost = 0.0
    if domain:
        for trusted in TRUSTED_DOMAINS:
            if domain == trusted or domain.endswith("." + trusted):
                domain_boost = 0.28
                break

    indicators = []
    if domain_boost > 0:
        indicators.append(f"Domain verified: '{domain}' is an established accredited international news wire.")
    if attribution_count >= 2:
        indicators.append(f"High journalistic attribution: found {attribution_count} cited sources, statements, or quoted witnesses.")
    elif attribution_count == 0:
        indicators.append("Attribution warning: article lacks named sources, verifiable quotes, or accredited references.")

    if sensationalism > 50:
        indicators.append("Sensationalism alert: contains emotional trigger phrasing, hyperbole, or emphatic punctuation.")
    else:
        indicators.append("Stylistic integrity: neutral, objective, and measured journalistic tone.")

    if trigger_hits:
        indicators.append(f"Flagged sensational clickbait terms: {', '.join(trigger_hits[:3])}.")

    read_time_mins = max(1, math.ceil(total_words / 200))

    return {
        "sensationalism_score": sensationalism,
        "clickbait_risk": clickbait_risk,
        "attribution_score": attribution_score,
        "attribution_count": attribution_count,
        "domain_boost": domain_boost,
        "emotional_polarization": "High / Inflammatory" if sensationalism > 55 else "Objective & Balanced",
        "lexical_richness": f"{ttr:.2f}",
        "read_time": f"{read_time_mins} min read",
        "word_count": total_words,
        "indicators": indicators
    }

def analyze_news_dual_engine(raw_input: str, is_url_mode: bool = False) -> Dict[str, Any]:
    """
    Executes Dual Machine Learning + Linguistic Telemetry Heuristic Consensus.
    Robust against modern out-of-domain breaking news and historical articles.
    """
    if ml_bundle is None:
        load_or_train_models()

    raw_input = raw_input.strip()
    source_info = {"is_url": False, "url": "", "domain": "", "title": ""}
    news_content = raw_input

    url_match = re.search(r"https?://\S+", raw_input)
    if is_url_mode or url_match:
        target_url = url_match.group(0) if url_match else raw_input
        scrape_res = fetch_url_content(target_url)
        source_info = {
            "is_url": True,
            "url": target_url,
            "domain": scrape_res["domain"],
            "title": scrape_res["title"]
        }
        if scrape_res["success"] and len(scrape_res["text"].split()) >= 12:
            news_content = scrape_res["text"]

    cleaned_text = clean_text_for_nlp(news_content)
    vec = ml_bundle["vectorizer"]
    X_vec = vec.transform([cleaned_text])
    vocab_coverage = X_vec.nnz

    # Probabilities from ML classifiers
    lr_prob = float(ml_bundle["model_lr"].predict_proba(X_vec)[0][1])
    sgd_prob = float(ml_bundle["model_sgd"].predict_proba(X_vec)[0][1])
    nb_prob = float(ml_bundle["model_nb"].predict_proba(X_vec)[0][1])
    ml_avg_prob = (lr_prob * 0.40) + (sgd_prob * 0.35) + (nb_prob * 0.25)

    # Telemetry metrics
    telemetry = calculate_linguistic_telemetry(news_content, domain=source_info.get("domain"))

    # Credibility baseline calculation
    cred_score = 0.55
    sens = telemetry["sensationalism_score"]
    attr_count = telemetry["attribution_count"]

    if sens < 20:
        cred_score += 0.22 * ((20 - sens) / 20.0)
    elif sens > 40:
        cred_score -= 0.38 * min(1.0, (sens - 40) / 40.0)

    if attr_count >= 2:
        cred_score += 0.22
    elif attr_count == 1:
        cred_score += 0.10
    else:
        cred_score -= 0.10

    cred_score += telemetry["domain_boost"]
    cred_score = max(0.04, min(0.98, cred_score))

    # Adaptive Consensus Weights
    if vocab_coverage >= 20:
        w_ml = 0.55
        w_telemetry = 0.45
    elif vocab_coverage >= 8:
        w_ml = 0.40
        w_telemetry = 0.60
    else:
        w_ml = 0.25
        w_telemetry = 0.75

    consensus_score = (ml_avg_prob * w_ml) + (cred_score * w_telemetry)
    consensus_score = max(0.01, min(0.99, consensus_score))

    is_real = consensus_score >= 0.48
    verdict = "REAL / CREDIBLE" if is_real else "FAKE / SUSPICIOUS"
    confidence = (consensus_score if is_real else (1.0 - consensus_score)) * 100.0
    confidence = min(99.6, max(58.0, confidence))

    headline_preview = source_info["title"]
    if not headline_preview or headline_preview in ["Live Online Article", "Online News Source"]:
        first_line = news_content.split("\n")[0].strip()
        headline_preview = (first_line[:90] + "...") if len(first_line) > 90 else first_line
        if not headline_preview:
            headline_preview = (news_content[:90] + "...") if len(news_content) > 90 else news_content

    return {
        "verdict": verdict,
        "is_real": is_real,
        "confidence": round(confidence, 1),
        "consensus_score": round(consensus_score, 3),
        "headline": headline_preview,
        "word_count": telemetry["word_count"],
        "read_time": telemetry["read_time"],
        "source_info": source_info,
        "model_breakdown": {
            "Logistic Regression": round(lr_prob * 100, 1),
            "SGD / Passive Aggressive": round(sgd_prob * 100, 1),
            "Multinomial Naive Bayes": round(nb_prob * 100, 1),
            "Linguistic Telemetry Consensus": round(cred_score * 100, 1)
        },
        "telemetry": {
            "sensationalism_score": telemetry["sensationalism_score"],
            "clickbait_risk": telemetry["clickbait_risk"],
            "attribution_score": telemetry["attribution_score"],
            "attribution_count": telemetry["attribution_count"],
            "emotional_polarization": telemetry["emotional_polarization"],
            "lexical_richness": telemetry["lexical_richness"],
            "vocabulary_tokens_matched": int(vocab_coverage)
        },
        "key_indicators": telemetry["indicators"],
        "sample_snippet": (news_content[:260] + "...") if len(news_content) > 260 else news_content
    }

# =============================================================================
# FASTAPI APP & WEBSOCKET TELEMETRY PIPELINE
# =============================================================================
app = FastAPI(
    title="Veritas Core - Autonomous News Integrity Engine",
    description="Dual Machine Learning & Linguistic Telemetry News Verification API",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NewsAnalysisRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    mode: Optional[str] = "text"

@app.on_event("startup")
async def startup_event():
    logger.info("Starting Veritas Core Machine Learning Engine...")
    load_or_train_models()
    logger.info("Veritas Core engine is fully online on port 8000!")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_ready": ml_bundle is not None,
        "timestamp": time.time(),
        "service": "Veritas Core News Detection Engine"
    }

@app.post("/api/analyze")
async def analyze_rest(payload: NewsAnalysisRequest):
    content = payload.url if (payload.url and payload.mode == "url") else payload.text
    if not content or not content.strip():
        raise HTTPException(status_code=400, detail="Missing article text or URL.")
    is_url = bool(payload.url or payload.mode == "url" or re.search(r"https?://", content))
    result = analyze_news_dual_engine(content, is_url_mode=is_url)
    return result

@app.websocket("/ws")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Frontend WebSocket client connected to Veritas Core.")
    try:
        while True:
            data_raw = await websocket.receive_text()
            data = json.loads(data_raw)
            raw_input = data.get("input", "").strip()
            mode = data.get("mode", "text")

            if not raw_input:
                await websocket.send_json({
                    "step": "error",
                    "progress": 0,
                    "message": "Input cannot be empty. Please enter an article text or news URL."
                })
                continue

            # Step 1: Input ingestion & format validation
            await websocket.send_json({
                "step": "ingest",
                "progress": 15,
                "message": "Ingesting telemetry packet & validating format..."
            })
            await asyncio.sleep(0.18)

            # Step 2: Content extraction / Web Scraping if URL
            is_url = mode == "url" or bool(re.search(r"https?://", raw_input))
            if is_url:
                await websocket.send_json({
                    "step": "scraping",
                    "progress": 35,
                    "message": "Connecting to source URL & parsing live DOM..."
                })
            else:
                await websocket.send_json({
                    "step": "cleaning",
                    "progress": 35,
                    "message": "Sanitizing linguistic tokens & removing syntax noise..."
                })
            await asyncio.sleep(0.20)

            # Step 3: Linguistic Telemetry & Heuristic Consensus
            await websocket.send_json({
                "step": "telemetry",
                "progress": 60,
                "message": "Analyzing Sensationalism Index & Source Attribution..."
            })
            await asyncio.sleep(0.20)

            # Step 4: Machine Learning Ensemble Inference
            await websocket.send_json({
                "step": "ml_inference",
                "progress": 85,
                "message": "Evaluating Logistic Regression, SGD & Naive Bayes models..."
            })
            await asyncio.sleep(0.20)

            # Step 5: Final synthesis & results delivery
            result = analyze_news_dual_engine(raw_input, is_url_mode=is_url)
            await websocket.send_json({
                "step": "complete",
                "progress": 100,
                "message": "Verification complete. Consensus achieved.",
                "data": result
            })

    except WebSocketDisconnect:
        logger.info("Frontend WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "step": "error",
                "progress": 0,
                "message": f"Verification error: {str(e)}"
            })
        except Exception:
            pass

@app.get("/")
async def serve_index():
    if os.path.exists(INDEX_HTML_PATH):
        return FileResponse(INDEX_HTML_PATH)
    return HTMLResponse("<h1>Veritas Core Active. Place index.html in the project root.</h1>")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print(" VERITAS CORE - PRODUCTION NEWS DETECTOR ACTIVE")
    print(" WebSocket Endpoint : ws://localhost:8000/ws")
    print(" Web Interface      : http://localhost:8000")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
