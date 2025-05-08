import numpy as np
import pandas as pd
import torch
from sklearn.cluster import DBSCAN
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import spacy
import re

device = 0 if torch.cuda.is_available() else -1
nlp = spacy.load("en_core_web_trf")
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda" if torch.cuda.is_available() else "cpu")

MAX_MODEL_TOKENS = 512


def extract_named_entities(text):
    doc = nlp(text)
    df = pd.DataFrame(
        [(e.text, e.label_, embedding_model.encode(e.text))
            for e in doc.ents
            if e.label_ in {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC"}],
            columns=['text', 'type', 'vec'])
    X = np.vstack(df.vec.to_numpy())
    dbscan = DBSCAN(metric='cosine', min_samples=1, eps=0.4)
    df['cluster'] = dbscan.fit_predict(X)
    groups = df.groupby(by=['cluster'])['text'].apply(list).to_dict()
    return groups

def get_context_windows(sentences, variants, window_size=1):
    windows = []
    for i, s in enumerate(sentences):
        if any(re.search(rf'\b{re.escape(v)}\b', s, flags=re.IGNORECASE)
               for v in variants):
            start = max(0, i - window_size)
            end   = min(len(sentences), i + window_size + 1)
            windows.append(" ".join(sentences[start:end]))
    return windows

def cluster_entities(entities):
    embeddings = embedding_model.encode(entities, convert_to_tensor=True)
    clusters = []
    used = set()

    for index, element in enumerate(entities):
        if element in used:
            continue
        cluster = [element]
        used.add(element)
        for second_index, second_element in enumerate(entities):
            if second_element in used:
                continue
            if util.pytorch_cos_sim(embeddings[index], embeddings[second_index]).item() >= 0.8:
                cluster.append(second_element)
                used.add(second_element)
        clusters.append(cluster)

    entity_map = {}
    for group in clusters:
        longest = group[0].strip()
        for variant in group:
            entity_map[variant] = longest

    return entity_map

def analyze_entity_sentiment(text):
    entities = extract_named_entities(text)
    sentences = sent_tokenize(text)

    all_windows = []
    window_counts = []
    for variants in entities.values():
        wins = get_context_windows(sentences, variants, window_size=1)
        all_windows.extend(wins)
        window_counts.append((variants[0], len(wins), variants))

    outputs = sentiment_model(all_windows, top_k=None)
    results = {}
    idx = 0
    for entity, count, variants in window_counts:
        entity_outs = outputs[idx: idx+count]
        idx += count

        pos = [score for out in entity_outs for score in (e["score"] for e in out if e["label"]=="POSITIVE")]
        neg = [score for out in entity_outs for score in (e["score"] for e in out if e["label"]=="NEGATIVE")]

        avg_pos, avg_neg = (np.mean(pos) if pos else 0, np.mean(neg) if neg else 0)
        if abs(avg_pos-avg_neg) < 0.1:
            sentiment = "NEUTRAL"
        else:
            sentiment = "POSITIVE" if avg_pos>avg_neg else "NEGATIVE"

        results[entity] = {
            "sentiment": sentiment,
            "score_diff": avg_pos-avg_neg,
            "mentions": count,
            "variants": variants
        }


    return results

