import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def to_list(x):
    if pd.isna(x):
        return []
    return [s.lower().strip() for s in str(x).replace(';', ',').split(',') if s.strip()]


def main():
    csv_path = r"C:\Users\HP\.cache\kagglehub\datasets\breejeshdhar\career-recommendation-dataset\versions\1\career_recommender.csv"
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Model')
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Update csv_path and re-run.")
        sys.exit(2)

    print('Loading dataset...')
    df = pd.read_csv(csv_path)

    # Rename columns (same mapping used in notebook)
    column_dict = {
        'What is your name?': 'Name',
        'What is your gender?': 'gender',
        'What was your course in UG?': 'course',
        'What is your UG specialization? Major Subject (Eg; Mathematics)': 'Specialization',
        'What are your interests?': 'interest',
        'What are your skills ? (Select multiple if necessary)': 'skills',
        'What was the average CGPA or Percentage obtained in under graduation?': 'grades',
        'Did you do any certification courses additionally?': 'Any_Add_Cert_Courses',
        'If yes, please specify your certificate course title.': 'Cert_Courses_Desc',
        'Are you working?': 'Working?',
        'If yes, then what is/was your first Job title in your current field of work? If not applicable, write NA.               ': 'Job_Title',
        'Have you done masters after undergraduation? If yes, mention your field of masters.(Eg; Masters in Mathematics)': 'Masters_Desc'
    }
    df = df.rename(columns=column_dict)

    # Drop unused columns
    drop_cols = ['Name', 'Specialization', 'Any_Add_Cert_Courses', 'Cert_Courses_Desc', 'Working?', 'Job_Title', 'Masters_Desc']
    for c in drop_cols:
        if c in df.columns:
            df.drop(c, axis=1, inplace=True)

    # Basic cleaning: fill null skills/interest with empty string
    if 'interest' in df.columns:
        df['interest'] = df['interest'].fillna('')
    if 'skills' in df.columns:
        df['skills'] = df['skills'].fillna('')

    # Normalize list columns
    if 'interest' in df.columns:
        df['interest'] = df['interest'].apply(to_list)
    if 'skills' in df.columns:
        df['skills'] = df['skills'].apply(to_list)

    # Label encode gender and target course
    le_gender = LabelEncoder()
    if 'gender' in df.columns:
        df['gender'] = df['gender'].fillna('unknown').astype(str)
        df['gender_enc'] = le_gender.fit_transform(df['gender'])
        joblib.dump(le_gender, os.path.join(model_dir, 'gender_le.pkl'))
        print('Saved gender_le.pkl')
    else:
        df['gender_enc'] = 0

    le_course = LabelEncoder()
    if 'course' not in df.columns:
        print('Course column not found, cannot train')
        sys.exit(3)
    df['course'] = df['course'].fillna('unknown').astype(str)
    y = le_course.fit_transform(df['course'])
    joblib.dump(le_course, os.path.join(model_dir, 'le.pkl'))
    print('Saved label encoder le.pkl (target)')

    # Fit MultiLabelBinarizer for interests and skills
    mlb_interest = MultiLabelBinarizer()
    mlb_skills = MultiLabelBinarizer()
    mlb_interest.fit(df['interest'])
    mlb_skills.fit(df['skills'])
    joblib.dump(mlb_interest, os.path.join(model_dir, 'mlb_interest.pkl'))
    joblib.dump(mlb_skills, os.path.join(model_dir, 'mlb.pkl'))
    print('Saved mlb_interest.pkl and mlb.pkl (skills)')

    # Build encoded DataFrame
    df_interest = pd.DataFrame(mlb_interest.transform(df['interest']), columns=mlb_interest.classes_)
    df_skills = pd.DataFrame(mlb_skills.transform(df['skills']), columns=mlb_skills.classes_)

    # Prepare base features (keep gender_enc and grades if exist)
    base_cols = []
    if 'gender_enc' in df.columns:
        base_cols.append('gender_enc')
    if 'grades' in df.columns:
        # try to coerce to numeric
        df['grades'] = pd.to_numeric(df['grades'], errors='coerce').fillna(0)
        base_cols.append('grades')

    X = pd.concat([df[base_cols].reset_index(drop=True), df_interest.reset_index(drop=True), df_skills.reset_index(drop=True)], axis=1)

    # Save feature columns order
    feature_cols = list(X.columns)
    joblib.dump(feature_cols, os.path.join(model_dir, 'feature_columns.pkl'))
    print(f'Saved feature_columns.pkl ({len(feature_cols)} columns)')

    # Scale numeric columns (for LR)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(model_dir, 'sc.sav'))
    print('Saved scaler sc.sav')

    # Train/test split (no stratify because some classes have very few samples)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=101)

    # Train RandomForestClassifier
    print('Training RandomForestClassifier...')
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=101, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print('RandomForestClassifier accuracy on test:', acc)
    print('Classification report (labels are encoded integers):')
    print(classification_report(y_test, y_pred))
    print('Label encoder class count:', len(le_course.classes_))
    joblib.dump(rf_clf, os.path.join(model_dir, 'rf_clf.sav'))
    print('Saved rf_clf.sav')

    # Train LogisticRegression (as alternative)
    print('Training LogisticRegression (multinomial, max_iter=1000)...')
    lr_clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
    lr_clf.fit(X_train, y_train)
    y_pred_lr = lr_clf.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print('LogisticRegression accuracy on test:', acc_lr)
    print('Classification report (LR) (labels are encoded integers):')
    print(classification_report(y_test, y_pred_lr))
    joblib.dump(lr_clf, os.path.join(model_dir, 'lr_clf.sav'))
    print('Saved lr_clf.sav')

    print('Retraining complete.')


if __name__ == '__main__':
    main()
