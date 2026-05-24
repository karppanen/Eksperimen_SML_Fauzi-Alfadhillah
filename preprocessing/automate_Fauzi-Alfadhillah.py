import argparse
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


FEATURE_COLUMNS = [
    "Age",
    "Total Salary (IDR)",
    "Total Spending (IDR)",
    "Frequency (Yearly)",
]
TARGET_COLUMN = "Membership Status"


def preprocess_data(input_path: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame:
    """Clean, encode, and scale the customer segmentation dataset."""
    input_path = Path(input_path)
    df = pd.read_csv(input_path)

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()

    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS).copy()

    label_encoder = LabelEncoder()
    df["membership_status_label"] = label_encoder.fit_transform(df[TARGET_COLUMN])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[FEATURE_COLUMNS])
    processed_df = pd.DataFrame(
        scaled_features,
        columns=[
            "age_scaled",
            "total_salary_idr_scaled",
            "total_spending_idr_scaled",
            "frequency_yearly_scaled",
        ],
    )
    processed_df["membership_status_label"] = df["membership_status_label"].to_numpy()
    processed_df["membership_status"] = df[TARGET_COLUMN].to_numpy()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(output_dir / "customer_segmentation_preprocessed.csv", index=False)

        mapping = pd.DataFrame(
            {
                "membership_status": label_encoder.classes_,
                "membership_status_label": range(len(label_encoder.classes_)),
            }
        )
        mapping.to_csv(output_dir / "label_mapping.csv", index=False)

    return processed_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Automate customer segmentation preprocessing.")
    parser.add_argument("--input", default="../CustomerSegmentation_raw.csv", help="Path to raw CSV file.")
    parser.add_argument(
        "--output-dir",
        default="CustomerSegmentation_preprocessing",
        help="Directory for preprocessed dataset files.",
    )
    args = parser.parse_args()

    processed_df = preprocess_data(args.input, args.output_dir)
    print(f"Preprocessing selesai. Shape data: {processed_df.shape}")
    print(f"Output tersimpan di: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
