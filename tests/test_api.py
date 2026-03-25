def _long_real_article() -> str:
    return (
        "Official government data according to a detailed report from the finance ministry shows steady "
        "employment growth across several states. The report explains that public records, audited figures, "
        "and district level data were reviewed over multiple months. Officials said the findings match "
        "independent surveys, and the document includes dates, named sources, and methodology for each "
        "section of the analysis so readers can verify the context without relying on rumors or anonymous posts."
    )


def _long_fake_article() -> str:
    return (
        "Breaking stories on the site promote a shocking secret treatment that insiders exposed without any "
        "official data, verifiable records, or transparent sourcing. The article repeats miracle language, "
        "claims hidden groups are blocking the truth, and pushes a cure narrative without named experts, "
        "documents, or public evidence. Readers are told the story was exposed by unknown sources and shared "
        "widely before anyone could review the facts."
    )


def test_root_renders_index_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.content_type


def test_root_head_returns_fast_ok(client):
    response = client.head("/")
    assert response.status_code == 200
    assert response.get_data(as_text=True) == ""


def test_root_returns_ok_when_template_missing(test_app, tmp_path):
    test_app.template_folder = str(tmp_path / "missing_templates")

    with test_app.test_client() as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.get_data(as_text=True) == "OK"
    assert "text/plain" in response.content_type


def test_version_route_returns_new_version(client):
    response = client.get("/version")
    assert response.status_code == 200
    assert response.get_data(as_text=True) == "NEW VERSION"


def test_predict_endpoint_success(client):
    response = client.post(
        "/predict",
        json={"text": _long_real_article(), "include_shap": False, "include_lime": False},
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["prediction"] in {"FAKE", "REAL", "UNCERTAIN"}
    assert "confidence" in payload
    assert "reason" in payload
    assert "top_fake_words" in payload
    assert "top_real_words" in payload
    assert isinstance(payload["top_fake_words"], list)
    assert isinstance(payload["top_real_words"], list)
    assert payload["top_fake_words"] or payload["top_real_words"]
    assert 40 <= payload["confidence"] <= 90


def test_predict_endpoint_validation(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["prediction"] == "INSUFFICIENT_CONTEXT"
    assert payload["confidence"] == 0
    assert "reason" in payload


def test_analyze_url_endpoint_success(client):
    response = client.post(
        "/analyze_url",
        json={"url": "https://example.com/news/story", "include_shap": False, "include_lime": False},
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["url"] == "https://example.com/news/story"
    assert payload["domain"] == "example.com"
    assert "article_preview" in payload
    assert payload["prediction"] in {"FAKE", "REAL", "UNCERTAIN"}
    assert "reason" in payload
    assert isinstance(payload["top_fake_words"], list)
    assert isinstance(payload["top_real_words"], list)
    assert payload["top_fake_words"] or payload["top_real_words"]
    assert 40 <= payload["confidence"] <= 90


def test_analyze_url_shortcuts_trusted_domains(test_app):
    test_app.config["ARTICLE_FETCHER"] = lambda _: _long_real_article()

    with test_app.test_client() as client:
        response = client.post("/analyze_url", json={"url": "https://www.bbc.com/news/world-123"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["domain"] == "bbc.com"
    assert payload["prediction"] == "REAL"
    assert payload["top_real_words"]
    assert "trusted_source" in payload["top_real_words"]


def test_analyze_url_shortcuts_fake_domains(test_app):
    test_app.config["ARTICLE_FETCHER"] = lambda _: _long_fake_article()

    with test_app.test_client() as client:
        response = client.post("/analyze_url", json={"url": "https://www.beforeitsnews.com/"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["domain"] == "beforeitsnews.com"
    assert payload["prediction"] == "FAKE"
    assert payload["top_fake_words"]
    assert "low_credibility_domain" in payload["top_fake_words"]


def test_analyze_url_returns_insufficient_context_for_social_domains(test_app):
    test_app.config["ARTICLE_FETCHER"] = lambda _: (_ for _ in ()).throw(AssertionError("extractor should not run"))

    with test_app.test_client() as client:
        response = client.post("/analyze_url", json={"url": "https://www.youtube.com/watch?v=123"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] == "INSUFFICIENT_CONTEXT"
    assert payload["confidence"] == 0
    assert payload["reason"] == "Social media content not suitable for analysis."


def test_analyze_url_returns_insufficient_context_for_non_article_url(client):
    response = client.post("/analyze_url", json={"url": "https://example.com/"})
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["prediction"] in {"REAL", "FAKE", "UNCERTAIN", "INSUFFICIENT_CONTEXT"}


def test_analyze_url_returns_uncertain_when_extraction_fails(test_app):
    test_app.config["ARTICLE_FETCHER"] = lambda _: {"prediction": "UNCERTAIN", "confidence": 40, "error": "Unable to extract article"}

    with test_app.test_client() as client:
        response = client.post("/analyze_url", json={"url": "https://example.com/news/story"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] == "UNCERTAIN"
    assert payload["confidence"] == 40
    assert payload["error"] == "Unable to extract article"
    assert payload["reason"] == "Unable to extract article"


def test_analyze_url_returns_uncertain_for_low_quality_extraction_text(test_app):
    test_app.config["ARTICLE_FETCHER"] = (
        lambda _: "Sign in to continue reading. Subscribe for newsletter access benefits and accept cookies. "
        "Login now to read more of this advertisement and unlock members only access benefits."
    )

    with test_app.test_client() as client:
        response = client.post("/analyze_url", json={"url": "https://example.com/news/story"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] == "UNCERTAIN"
    assert payload["confidence"] == 45
    assert payload["error"] == "Low quality extracted content"
    assert payload["article_char_count"] > 0


def test_analytics_endpoint(client):
    client.post("/predict", json={"text": "The report is true.", "include_shap": False, "include_lime": False})
    response = client.get("/api/analytics")
    assert response.status_code == 200
    payload = response.get_json()
    assert "total_analyzed" in payload
    assert "prediction_distribution" in payload
