"""End-to-end test for the complete review-analysis workflow."""

from pathlib import Path

from analysis_functions import run_analysis

SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "sample_reviews.csv"


def test_complete_analysis_workflow():
    """Load, preprocess, filter, summarize, and prepare ML data."""
    results = run_analysis(SAMPLE_DATA_PATH)

    reviews = results["reviews"]
    helpful_reviews = results["helpful_positive_reviews"]
    score_summary = results["score_summary"]
    ml_data = results["ml_data"]

    # Data was loaded and preprocessed.
    assert len(reviews) == 5
    assert "ReviewDate" in reviews.columns

    # Filtering selected the expected reviews.
    assert helpful_reviews["Id"].tolist() == [1, 2]

    # Grouping accounted for every input review.
    assert score_summary["ReviewCount"].sum() == 5

    # Neutral reviews were excluded from ML preparation.
    assert len(ml_data) == 4
    assert 3 not in ml_data["Score"].values
