import pandas as pd

from app.preprocessing import deduplicate_dataframe, deduplicate_texts, preprocess_text


def test_preprocess_text_pipeline():
    text = "Visit https://example.com! CATS, dogs, and running runners."
    processed = preprocess_text(text)

    assert "https" not in processed
    assert "," not in processed
    assert "!" not in processed
    assert "and" not in processed
    assert processed == processed.lower()
    assert "cat" in processed or "cats" in processed


def test_deduplicate_texts_preserves_order():
    values = ["alpha", "beta", "alpha", "gamma", "beta"]
    assert deduplicate_texts(values) == ["alpha", "beta", "gamma"]


def test_deduplicate_dataframe():
    df = pd.DataFrame({"processed_text": ["a", "b", "a"], "label": [0, 1, 0]})
    result = deduplicate_dataframe(df, text_column="processed_text")
    assert len(result) == 2

