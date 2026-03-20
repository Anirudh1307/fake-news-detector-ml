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
        json={"text": "Verified records support the policy claim.", "include_shap": False, "include_lime": False},
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["prediction"].startswith(("FAKE", "REAL"))
    assert "confidence" in payload
    assert "reason" in payload
    assert "top_fake_words" in payload
    assert "top_real_words" in payload
    assert isinstance(payload["top_fake_words"], list)
    assert isinstance(payload["top_real_words"], list)
    assert payload["top_fake_words"] or payload["top_real_words"]
    assert payload["confidence"] <= 95


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
    assert payload["prediction"].startswith(("FAKE", "REAL"))
    assert "reason" in payload
    assert isinstance(payload["top_fake_words"], list)
    assert isinstance(payload["top_real_words"], list)
    assert payload["top_fake_words"] or payload["top_real_words"]


def test_analyze_url_shortcuts_trusted_domains(test_app):
    test_app.config["ARTICLE_FETCHER"] = lambda _: (_ for _ in ()).throw(AssertionError("extractor should not run"))

    with test_app.test_client() as client:
        response = client.post("/analyze_url", json={"url": "https://www.bbc.com/news/world-123"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] == "REAL"
    assert payload["confidence"] == 85
    assert payload["domain"] == "bbc.com"
    assert payload["top_real_words"] == ["trusted_source"]
    assert payload["top_fake_words"] == []


def test_analyze_url_shortcuts_fake_domains(test_app):
    test_app.config["ARTICLE_FETCHER"] = lambda _: (_ for _ in ()).throw(AssertionError("extractor should not run"))

    with test_app.test_client() as client:
        response = client.post("/analyze_url", json={"url": "https://www.beforeitsnews.com/"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] == "FAKE"
    assert payload["confidence"] == 80
    assert payload["domain"] == "beforeitsnews.com"
    assert payload["top_fake_words"] == ["low_credibility_domain"]
    assert payload["top_real_words"] == []


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
    assert payload["prediction"] == "INSUFFICIENT_CONTEXT"
    assert payload["confidence"] == 0


def test_analyze_url_returns_insufficient_context_when_extraction_fails(test_app):
    test_app.config["ARTICLE_FETCHER"] = lambda _: ""

    with test_app.test_client() as client:
        response = client.post("/analyze_url", json={"url": "https://example.com/news/story"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"] == "INSUFFICIENT_CONTEXT"
    assert payload["confidence"] == 0
    assert "reason" in payload


def test_analyze_url_allows_shorter_but_usable_extraction_text(test_app):
    test_app.config["ARTICLE_FETCHER"] = lambda _: "This report cites named sources and supporting facts."

    with test_app.test_client() as client:
        response = client.post("/analyze_url", json={"url": "https://example.com/news/story"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["prediction"].startswith(("FAKE", "REAL"))
    assert payload["article_char_count"] >= 30


def test_analytics_endpoint(client):
    client.post("/predict", json={"text": "The report is true.", "include_shap": False, "include_lime": False})
    response = client.get("/api/analytics")
    assert response.status_code == 200
    payload = response.get_json()
    assert "total_analyzed" in payload
    assert "prediction_distribution" in payload
