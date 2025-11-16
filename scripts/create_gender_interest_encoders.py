import os
import sys
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer


def to_list(x):
    if pd.isna(x):
        return []
    return [s.lower().strip() for s in str(x).replace(';', ',').split(',') if s.strip()]


def main():
    csv_path = r"C:\Users\HP\.cache\kagglehub\datasets\breejeshdhar\career-recommendation-dataset\versions\1\career_recommender.csv"
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Model')
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Please update the path in this script.")
        sys.exit(2)

    df = pd.read_csv(csv_path)

    # Gender encoder
    gender_col = 'What is your gender?'
    if gender_col in df.columns:
        le = LabelEncoder()
        df_gender = df[gender_col].fillna('unknown').astype(str)
        le.fit(df_gender)
        joblib.dump(le, os.path.join(model_dir, 'gender_le.pkl'))
    else:
        print(f'Gender column {gender_col} not found in CSV; skipping gender encoder')

    # Interests mlb
    interest_col = 'What are your interests?'
    if interest_col in df.columns:
        df_interest = df[interest_col].fillna('').apply(to_list)
        mlb_interest = MultiLabelBinarizer()
        mlb_interest.fit(df_interest)
        joblib.dump(mlb_interest, os.path.join(model_dir, 'mlb_interest.pkl'))
    else:
        print(f'Interest column {interest_col} not found in CSV; skipping interest mlb')

    # Optionally ensure skills mlb exists (recreate if missing)
    skills_col = 'What are your skills ? (Select multiple if necessary)'
    if skills_col in df.columns:
        df_skills = df[skills_col].fillna('').apply(to_list)
        mlb_skills = MultiLabelBinarizer()
        mlb_skills.fit(df_skills)
        joblib.dump(mlb_skills, os.path.join(model_dir, 'mlb.pkl'))


if __name__ == '__main__':
    main()
