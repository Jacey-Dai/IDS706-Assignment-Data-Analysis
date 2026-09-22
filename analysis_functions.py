"""Reusable functions for the Amazon food review analysis."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "Id",
    "Score",
    "Time",
    "Text",
    "HelpfulnessNumerator",
    "HelpfulnessDenominator",
}


def load_reviews(file_path):
    """Load review data from a CSV file and validate required columns."""
    file_path = Path(file_path)
    reviews = pd.read_csv(file_path)

    missing_columns = REQUIRED_COLUMNS - set(reviews.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing_names}")

    return reviews


def preprocess_reviews(reviews):
    """Clean review text and convert Unix timestamps to dates."""
    processed = reviews.copy()

    processed["ReviewDate"] = pd.to_datetime(
        processed["Time"],
        unit="s",
    )

    processed["Text"] = (
        processed["Text"]
        .fillna("")
        .str.replace(r"<br\s*/?>", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return processed


def add_helpfulness_ratio(reviews):
    """Add a helpfulness ratio while avoiding division by zero."""
    result = reviews.copy()

    valid_denominator = result["HelpfulnessDenominator"].where(
        result["HelpfulnessDenominator"] > 0
    )

    result["HelpfulnessRatio"] = result["HelpfulnessNumerator"] / valid_denominator

    return result


def filter_helpful_positive_reviews(
    reviews,
    minimum_votes=10,
    minimum_score=4,
    minimum_ratio=0.80,
):
    """Select positive reviews supported by sufficient helpfulness votes."""
    reviews_with_ratio = add_helpfulness_ratio(reviews)

    filtered = reviews_with_ratio[
        (reviews_with_ratio["HelpfulnessDenominator"] >= minimum_votes)
        & (reviews_with_ratio["Score"] >= minimum_score)
        & (reviews_with_ratio["HelpfulnessRatio"] >= minimum_ratio)
    ].copy()

    return filtered


def summarize_scores(reviews):
    """Group reviews by score and calculate summary statistics."""
    summary = (
        reviews.groupby("Score")
        .agg(
            ReviewCount=("Id", "count"),
            AverageHelpfulVotes=(
                "HelpfulnessNumerator",
                "mean",
            ),
            AverageTotalVotes=(
                "HelpfulnessDenominator",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("Score")
        .reset_index(drop=True)
    )

    summary["Percentage"] = summary["ReviewCount"] / summary["ReviewCount"].sum() * 100

    return summary


def prepare_ml_data(reviews):
    """Create cleaned text and binary labels for sentiment modeling."""
    ml_data = reviews.loc[
        reviews["Score"] != 3,
        ["Text", "Score"],
    ].copy()

    ml_data["Text"] = (
        ml_data["Text"]
        .fillna("")
        .str.replace(r"<br\s*/?>", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    ml_data["Positive"] = (ml_data["Score"] >= 4).astype(int)

    return ml_data


def run_analysis(file_path):
    """Run the core analysis workflow from CSV loading to summaries."""
    reviews = load_reviews(file_path)
    processed_reviews = preprocess_reviews(reviews)

    helpful_positive_reviews = filter_helpful_positive_reviews(processed_reviews)

    score_summary = summarize_scores(processed_reviews)

    ml_data = prepare_ml_data(processed_reviews)

    return {
        "reviews": processed_reviews,
        "helpful_positive_reviews": helpful_positive_reviews,
        "score_summary": score_summary,
        "ml_data": ml_data,
    }
