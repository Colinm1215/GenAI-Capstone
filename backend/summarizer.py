from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
import nltk
nltk.download("punkt")

MAX_MODEL_TOKENS = 512

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer_pipeline = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=-1
)

def generate_summary(prompt, max_tokens=MAX_MODEL_TOKENS):
    token_count = len(prompt.split())
    safe_max = min(max_tokens, token_count) if token_count < max_tokens else max_tokens
    result = summarizer_pipeline(
        prompt,
        max_new_tokens=safe_max,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        length_penalty=2.0,
        early_stopping=True
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
