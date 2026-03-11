from __future__ import annotations

from collections import Counter


def build_analytics_summary(records: list[dict]) -> dict:
    if not records:
        return {
            "total_analyzed": 0,
            "prediction_distribution": {"FAKE NEWS": 0, "REAL NEWS": 0},
            "top_fake_keywords": [],
            "confidence_histogram": [],
        }

    prediction_counts = Counter(record.get("prediction", "UNKNOWN") for record in records)
    confidences = [float(record.get("confidence", 0)) for record in records]

    fake_keywords = Counter()
    for record in records:
        if record.get("prediction") != "FAKE NEWS":
            continue
        words = (
            record.get("explanation", {})
            .get("top_fake_words", [])
        )
        for item in words:
            word = item.get("word")
            if word:
                fake_keywords[word] += 1

    bins = [0, 20, 40, 60, 80, 100]
    histogram = []
    for start, end in zip(bins[:-1], bins[1:]):
        count = sum(start <= score < end for score in confidences)
        histogram.append({"range": f"{start}-{end}", "count": count})
    histogram[-1]["count"] = sum(80 <= score <= 100 for score in confidences)

    return {
        "total_analyzed": len(records),
        "prediction_distribution": {
            "FAKE NEWS": prediction_counts.get("FAKE NEWS", 0),
            "REAL NEWS": prediction_counts.get("REAL NEWS", 0),
        },
        "top_fake_keywords": [
            {"word": word, "count": count} for word, count in fake_keywords.most_common(10)
        ],
        "confidence_histogram": histogram,
    }

