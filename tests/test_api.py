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
    assert payload["prediction"] in {"FAKE NEWS", "REAL NEWS"}
    assert "confidence" in payload
    assert "top_fake_words" in payload
    assert "top_real_words" in payload


def test_predict_endpoint_validation(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400


def test_analyze_url_endpoint_success(client):
    response = client.post(
        "/analyze_url",
        json={"url": "https://example.com/story", "include_shap": False, "include_lime": False},
    )
    assert response.status_code == 200

    payload = response.get_json()
    assert payload["url"] == "https://example.com/story"
    assert "article_preview" in payload
    assert payload["prediction"] in {"FAKE NEWS", "REAL NEWS"}


def test_analytics_endpoint(client):
    client.post("/predict", json={"text": "The report is true.", "include_shap": False, "include_lime": False})
    response = client.get("/api/analytics")
    assert response.status_code == 200
    payload = response.get_json()
    assert "total_analyzed" in payload
    assert "prediction_distribution" in payload
