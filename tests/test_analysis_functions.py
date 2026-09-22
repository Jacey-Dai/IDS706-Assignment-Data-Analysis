"""Unit tests for the Amazon review analysis functions."""

from pathlib import Path

import pandas as pd
import pytest

from analysis_functions import (
    add_helpfulness_ratio,
    filter_helpful_positive_reviews,
    load_reviews,
    prepare_ml_data,
    preprocess_reviews,
    summarize_scores,
)

SAMPLE_DATA_PATH = Path(__file__).parent / "data" / "sample_reviews.csv"


def test_load_reviews():
    """The loader should read the sample CSV with all rows."""
    reviews = load_reviews(SAMPLE_DATA_PATH)

    assert len(reviews) == 5
    assert "Score" in reviews.columns
    assert "Text" in reviews.columns


def test_load_reviews_rejects_missing_columns(tmp_path):
    """The loader should reject data missing required columns."""
    invalid_path = tmp_path / "invalid.csv"

    invalid_data = pd.DataFrame(
        {
            "Id": [1],
            "Score": [5],
        }
    )
    invalid_data.to_csv(invalid_path, index=False)

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        load_reviews(invalid_path)


def test_preprocess_reviews():
    """Preprocessing should create dates and clean HTML breaks."""
    reviews = load_reviews(SAMPLE_DATA_PATH)
    processed = preprocess_reviews(reviews)

    assert "ReviewDate" in processed.columns
    assert pd.api.types.is_datetime64_any_dtype(processed["ReviewDate"])
    assert "<br" not in processed.loc[0, "Text"]
    assert processed.loc[0, "Text"] == "Great product Would buy again."


def test_helpfulness_ratio_handles_zero_denominator():
    """A zero denominator should produce a missing ratio."""
    reviews = load_reviews(SAMPLE_DATA_PATH)
    result = add_helpfulness_ratio(reviews)

    assert result.loc[0, "HelpfulnessRatio"] == 0.9
    assert result.loc[1, "HelpfulnessRatio"] == 0.8
    assert pd.isna(result.loc[3, "HelpfulnessRatio"])


def test_filter_helpful_positive_reviews():
    """Only sufficiently helpful positive reviews should remain."""
    reviews = load_reviews(SAMPLE_DATA_PATH)

    filtered = filter_helpful_positive_reviews(reviews)

    assert filtered["Id"].tolist() == [1, 2]
    assert (filtered["HelpfulnessDenominator"] >= 10).all()
    assert (filtered["Score"] >= 4).all()
    assert (filtered["HelpfulnessRatio"] >= 0.80).all()


def test_summarize_scores():
    """Score summaries should contain correct counts and percentages."""
    reviews = load_reviews(SAMPLE_DATA_PATH)
    summary = summarize_scores(reviews)

    assert summary["ReviewCount"].sum() == 5
    assert set(summary["Score"]) == {
        1,
        2,
        3,
        4,
        5,
    }
    assert summary["Percentage"].sum() == pytest.approx(100.0)
