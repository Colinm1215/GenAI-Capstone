import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
import nltk
import os

os.environ["NLTK_DATA"] = "/tmp/nltk_data"
nltk.download("punkt", download_dir="/tmp/nltk_data")
nltk.download("punkt_tab", download_dir="/tmp/nltk_data")
nltk.data.path.append("/tmp/nltk_data")

MAX_MODEL_TOKENS = 512
device = 0 if torch.cuda.is_available() else -1

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer_pipeline = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=device
)

def generate_summary(prompt, max_tokens=MAX_MODEL_TOKENS):
    tokens = tokenizer(prompt, return_tensors="pt").input_ids[0]
    input_len = tokens.shape[-1]
    summary_len = min(int(input_len * 0.6), max_tokens)

    result = summarizer_pipeline(
        prompt,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        early_stopping=True,
        max_length=summary_len
    )
    return result[0]["summary_text"].strip()

def summarize_text(text):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_len = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())
        if sentence_len == 0:
            continue

        if sentence_len > MAX_MODEL_TOKENS:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_chunk_len = 0
            chunks.append([sentence])
            continue

        if current_chunk_len + sentence_len <= MAX_MODEL_TOKENS:
            current_chunk.append(sentence)
            current_chunk_len += sentence_len
        else:
            chunks.append(current_chunk)
            current_chunk = [sentence]
            current_chunk_len = sentence_len

    if current_chunk:
        chunks.append(current_chunk)

    chunk_summaries = []
    for chunk in chunks:
        text_chunk = " ".join(chunk)
        try:
            summary = generate_summary(text_chunk)
            chunk_summaries.append(summary)
        except Exception as e:
            print(f"Error : {e}")
            chunk_summaries.append(text_chunk)

    if len(chunk_summaries) > 1:
        combined_chunks = " ".join(chunk_summaries)
        return combined_chunks

    return chunk_summaries[0]
