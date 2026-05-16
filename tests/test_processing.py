from app.schemas.paper import PaperCreate
from app.schemas.validation import IdeaSubmit, ValidationResponse
from app.utils.processing import strip_noise


def test_strip_noise_removes_urls_dois_and_citations():
    text = (
        "See https://example.com/paper and DOI 10.1234/ABC.DEF "
        "for prior work [1, 2]. Keep this sentence."
    )

    assert strip_noise(text) == "See and DOI for prior work . Keep this sentence."


def test_schema_validation_accepts_expected_payloads():
    paper = PaperCreate(
        external_id=10,
        title="Arabic NLP Search",
        abstract="A multilingual retrieval study.",
    )
    idea = IdeaSubmit(
        title="New idea",
        abstract="Validate this idea",
        keywords="nlp, search",
    )
    response = ValidationResponse(is_novel=True, message="ok")

    assert paper.keywords is None
    assert idea.keywords == "nlp, search"
    assert response.similar_papers == []
