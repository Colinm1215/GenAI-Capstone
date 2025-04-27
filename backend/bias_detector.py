import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import spacy
import re

nlp = spacy.load("en_core_web_trf")
tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=-1)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

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
    results = {}
    sentences = sent_tokenize(text.lower())

    for id, variants in entities.items():
        entity = variants[0]
        related_sentences = [s for s in sentences
                             if any(re.search(rf'\b{re.escape(v)}\b', s, flags=re.IGNORECASE)
                             for v in variants)]

        if not related_sentences:
            continue

        chunks = []
        current = []
        for s in related_sentences:
            if len(" ".join(current + [s]).split()) > MAX_MODEL_TOKENS:
                chunks.append(" ".join(current))
                current = [s]
            else:
                current.append(s)
        if current:
            chunks.append(" ".join(current))

        sentiments = []
        for chunk in chunks:
            try:
                input_ids = tokenizer.encode(chunk, truncation=True, max_length=MAX_MODEL_TOKENS, return_tensors=None)
                out = sentiment_model(tokenizer.decode(input_ids, skip_special_tokens=True))
                sentiments.append(out[0]["label"])
            except Exception as e:
                print(f"Error for : '{entity}' : {e}")
                continue

        if sentiments:
            count = Counter(sentiments)
            final_sentiment = count.most_common(1)[0][0]
            results[entity] = {
                "sentiment": final_sentiment,
                "details": count,
                "chunks_evaluated": len(sentiments),
                "variants": variants
            }

    return results
