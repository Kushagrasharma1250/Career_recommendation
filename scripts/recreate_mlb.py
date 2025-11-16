import os
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def main():
    # Path to dataset used in the notebook
    csv_path = r"C:\Users\HP\.cache\kagglehub\datasets\breejeshdhar\career-recommendation-dataset\versions\1\career_recommender.csv"
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Model')
    os.makedirs(model_dir, exist_ok=True)
    mlb_path = os.path.join(model_dir, 'mlb.pkl')

    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Please update the path in this script.")
        sys.exit(2)

    print(f"Loading dataset from {csv_path} ...")
    df = pd.read_csv(csv_path)

    # Column name observed in the notebook
    col = 'What are your skills ? (Select multiple if necessary)'
    if col not in df.columns:
        print(f"Expected column '{col}' not found in CSV. Available columns: {list(df.columns)[:10]}")
        sys.exit(3)

    # Normalize and split into lists (same processing as notebook)
    def to_list(x):
        if pd.isna(x):
            return []
        return [s.lower().strip() for s in str(x).replace(';', ',').split(',') if s.strip()]

    df['skills_list'] = df[col].apply(to_list)

    mlb = MultiLabelBinarizer()
    print("Fitting MultiLabelBinarizer on skills ...")
    mlb.fit(df['skills_list'])

    joblib.dump(mlb, mlb_path)


if __name__ == '__main__':
    main()
