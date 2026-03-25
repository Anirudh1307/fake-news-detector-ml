import app.utils as utils


def _long_article_text() -> str:
    return (
        "Official government data according to a detailed report from the finance ministry shows steady "
        "employment growth across several states. The report explains that public records, audited figures, "
        "and district level data were reviewed over multiple months. Officials said the findings match "
        "independent surveys, and the document includes dates, named sources, and methodology for each "
        "section of the analysis so readers can verify the context without relying on rumors or anonymous posts."
    )


def test_fetch_article_text_falls_back_after_low_quality_trafilatura(monkeypatch):
    monkeypatch.setattr(utils, "_download_html", lambda _: "<html></html>")
    monkeypatch.setattr(
        utils,
        "_extract_with_trafilatura",
        lambda url, html=None: (
            "Sign in to continue reading this page. Subscribe for newsletter access benefits and accept cookies. "
            "Login is required to continue, and this advertisement repeats access benefits for members only."
        ),
    )
    monkeypatch.setattr(utils, "_extract_with_newspaper", lambda _: _long_article_text())
    monkeypatch.setattr(
        utils,
        "_extract_with_bs4",
        lambda url, html=None: (_ for _ in ()).throw(AssertionError("bs4 should not run after strong newspaper text")),
    )

    result = utils.fetch_article_text("https://example.com/news/story")

    assert isinstance(result, str)
    assert "Official government data" in result


def test_fetch_article_text_returns_low_quality_error_when_all_extractors_are_weak(monkeypatch):
    weak_text = (
        "Sign in to continue reading. Subscribe for newsletter access benefits and accept cookies. "
        "Login now to keep reading this advertisement and unlock access benefits for members only."
    )
    monkeypatch.setattr(utils, "_download_html", lambda _: "<html></html>")
    monkeypatch.setattr(utils, "_extract_with_trafilatura", lambda url, html=None: weak_text)
    monkeypatch.setattr(utils, "_extract_with_newspaper", lambda _: weak_text)
    monkeypatch.setattr(utils, "_extract_with_bs4", lambda url, html=None: weak_text)

    result = utils.fetch_article_text("https://example.com/news/story")

    assert result == {
        "prediction": "UNCERTAIN",
        "confidence": 45,
        "error": "Low quality extracted content",
    }


def test_fetch_article_text_returns_unable_to_extract_when_all_extractors_fail(monkeypatch):
    monkeypatch.setattr(utils, "_download_html", lambda _: "<html></html>")
    monkeypatch.setattr(utils, "_extract_with_trafilatura", lambda url, html=None: "")
    monkeypatch.setattr(utils, "_extract_with_newspaper", lambda _: "")
    monkeypatch.setattr(utils, "_extract_with_bs4", lambda url, html=None: "")

    result = utils.fetch_article_text("https://example.com/news/story")

    assert result == {
        "prediction": "UNCERTAIN",
        "confidence": 40,
        "error": "Unable to extract article",
    }
