import os
import re
import time

import evaluate

from backend.summarizer import summarize_text
from backend.bias_detector import analyze_entity_sentiment

rouge = evaluate.load("rouge")

alias_fallback = {
    "the washington post’s": "washington post",
    "bezos": "jeff bezos",
    "kkk": "kkk",
    "klan": "kkk"
}

def clean_entity(entity):
    e = entity.lower().strip()
    e = re.sub(r"'s$", "", e)
    e = re.sub(r"^the\s+", "", e)
    e = re.sub(r"\s*\(.*?\)", "", e)
    e = re.sub(r"[^\w\s-]", "", e)
    return e


def match_entities(pred, ref):
    matched = {}
    unmatched = {}

    norm_reference = {clean_entity(k): v.lower() for k, v in ref.items()}
    found = []
    for raw_entity, data in pred.items():
        sentiment = data["sentiment"].lower()
        variants = [clean_entity(v) for v in data.get("variants", [])]
        entity_clean = clean_entity(raw_entity)
        entity_clean = alias_fallback.get(entity_clean, entity_clean)
        found.extend([raw_entity] + variants)

        if entity_clean in norm_reference:
            matched[raw_entity] = sentiment == norm_reference[entity_clean]
            continue

        found_match = False
        for var in variants:
            if var in norm_reference:
                matched[raw_entity] = sentiment == norm_reference[var]
                found_match = True
                break

        if not found_match:
            unmatched[raw_entity] = False

    for ref_raw in norm_reference:
        ref_clean = clean_entity(ref_raw)
        already_matched = any(
            clean_entity(m) == ref_clean or any(clean_entity(v) == ref_clean for v in pred.get(m, {}).get("variants", []))
            for m in matched
        )
        if not already_matched:
            unmatched[ref_raw] = False

    true_positives = len([k for k, v in matched.items() if v is True])
    false_positives = len([k for k, v in matched.items() if v is False])
    return true_positives, false_positives, len(matched), len(unmatched)

def evaluate_summaries(prediction, reference):
    return rouge.compute(predictions=prediction, references=reference)


def evaluate_bias(pred_bias, ref_bias):
    total_ref = sum(len(r) for r in ref_bias)
    correct = 0
    false_neg = 0
    for i in range(len(ref_bias)):
        ref_map = { clean_entity(k): v.lower() for k,v in ref_bias[i].items() }
        pred_map = {}
        for raw_ent, data in pred_bias[i].items():
            key = clean_entity(raw_ent)
            pred_map[key] = data["sentiment"].lower()

            for var in data.get("variants", []):
                pred_map[clean_entity(var)] = data["sentiment"].lower()

        for ref_ent_clean, true_sent in ref_map.items():
            ref_ent_clean = re.sub(r"\s*\(.*?\)", "", ref_ent_clean)
            pred_sent = pred_map.get(ref_ent_clean)
            if pred_sent is None:
                false_neg += 1
            elif pred_sent == true_sent:
                correct += 1

    accuracy = correct / total_ref if total_ref else 0
    return accuracy, correct, total_ref


if __name__ == "__main__":
    pred_summaries = []
    ref_summaries = [
        "Patrick Soon-Shiong, owner of the LA Times, has implemented AI tools that label opinion pieces with political bias and force the inclusion of opposing views—especially to balance criticism of Donald Trump. Critics argue this undermines editorial independence and leans toward promoting Trump-aligned narratives. The AI system made controversial inclusions like defenses of the KKK and pro-Trump reinterpretations. Many journalists have left, and industry voices criticize the trend of media owners yielding to political pressures.",
        "Senator Chris Van Hollen criticized the Trump administration for deporting Kilmar Abrego Garcia despite a court ruling barring his removal. Garcia, now detained in El Salvador, was allegedly falsely labeled a gang member without evidence. Van Hollen emphasizes the constitutional risk posed by denying due process, warning that it jeopardizes rights for all. The administration has ignored court facilitation orders and allegedly provided no instructions to the embassy. The case is framed as both a human rights and legal crisis.",
        "The Israeli military admitted to professional misconduct in the killing of 15 rescue workers in Gaza, including UN staff and paramedics, and dismissed a deputy commander. However, the military ruled out criminal charges and conducted the investigation internally. This has drawn sharp criticism from human rights lawyers and international observers, who argue it reflects ongoing impunity and undermines justice. The event has intensified scrutiny of Israel’s conduct in Gaza and raised calls for independent war crimes investigations.",
    ]
    pred_bias = []
    ref_bias = [
        {
            "Patrick Soon-Shiong": "negative",
            "Los Angeles Times": "negative",
            "Donald Trump": "negative",
            "Kamala Harris": "positive",
            "Lois Beckett": "positive",
            "LA Guild": "positive",
            "Ryan Mac": "positive",
            "ABC News": "negative",
            "CBS": "negative",
            "Disney": "negative",
            "Paramount": "negative",
            "Washington Post": "negative",
            "Jeff Bezos": "negative",
            "Axios": "positive",
            "Sara Fischer": "positive"
        },
        {
            "Chris Van Hollen": "positive",
            "Kilmar Abrego Garcia": "positive",
            "Donald Trump": "negative",
            "CECOT prison": "negative",
            "Gavin Newsom": "negative",
            "Dana Bash": "positive",
            "MS-13": "negative",
            "US Embassy": "negative",
            "El Salvador": "negative"
        },
        {
            "IDF": "negative",
            "Golani Brigade": "negative",
            "Itamar Ben-Gvir": "negative",
            "Hamas": "negative",
            "Palestinian Red Crescent Society": "positive",
            "United Nations (UN)": "positive",
            "Sawsan Zaher": "positive",
            "Daniel Machover": "positive",
            "Ahmed Dhair": "positive",
            "Yoav Har-Even": "negative",
            "Ziv Stahl": "positive",
            "Yesh Din": "positive",
            "Benjamin Netanyahu": "negative",
            "Yoav Gallant": "negative",
            "ICC": "positive"
        }
    ]

    files = ["1.txt", "2.txt", "3.txt"]

    summary_times = []
    sentiment_times = []

    for i in range(len(files)):
        file = files[i]
        path = os.path.join(os.path.dirname(__file__), file)
        with open(file, "r", encoding="utf-8") as f:
            file_text = f.read()
            start_time = time.time()
            summary = summarize_text(file_text)
            summary_duration = time.time() - start_time
            summary_times.append(summary_duration)

            pred_summaries.append(summary)

            start_time = time.time()
            bias = analyze_entity_sentiment(file_text)
            sentiment_duration = time.time() - start_time
            sentiment_times.append(sentiment_duration)

            pred_bias.append(bias)


    summary_results = evaluate_summaries(pred_summaries, ref_summaries)
    print(f"""
    Rouge Evaluation Results:
    ROUGE-1 : {summary_results["rouge1"]}
    ROUGE-2 : {summary_results["rouge2"]}
    ROUGE-L : {summary_results["rougeL"]}
    """)

    accuracy, matched, total = evaluate_bias(pred_bias, ref_bias)
    print(f"""
    Bias Evaluation Results:
    Bias Accuracy : {accuracy}
    Matched Entities : {matched}
    Total Entities : {total}
    """)

    average_summary_inference = sum(summary_times) / len(summary_times)
    average_sentiment_inference = sum(sentiment_times) / len(sentiment_times)
    print(f"""
    Average Inference Times: {average_summary_inference:.2f} seconds for summaries,
                             {average_sentiment_inference:.2f} seconds for sentiment analysis
    """)


