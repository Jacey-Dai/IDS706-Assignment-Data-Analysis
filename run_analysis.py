"""Command-line entry point for running the review analysis."""

import argparse

from analysis_functions import run_analysis


def main():
    """Run the analysis and print a concise summary."""
    parser = argparse.ArgumentParser(
        description="Analyze an Amazon food reviews CSV file."
    )
    parser.add_argument(
        "--data",
        default="tests/data/sample_reviews.csv",
        help="Path to the reviews CSV file.",
    )
    args = parser.parse_args()

    results = run_analysis(args.data)

    reviews = results["reviews"]
    helpful_reviews = results["helpful_positive_reviews"]
    score_summary = results["score_summary"]
    ml_data = results["ml_data"]

    print("Amazon Fine Food Reviews Analysis")
    print(f"Total reviews: {len(reviews)}")
    print(f"Highly helpful positive reviews: {len(helpful_reviews)}")
    print(f"Reviews available for ML: {len(ml_data)}")
    print("\nScore summary:")
    print(score_summary.to_string(index=False))


if __name__ == "__main__":
    main()
