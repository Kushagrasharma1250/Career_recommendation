from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

app = Flask(__name__)

# Load components (use `Model/` directory)
le = joblib.load('Model/le.pkl')
mlb = joblib.load('Model/mlb.pkl')
scaler = joblib.load('Model/sc.sav')
lr_model = joblib.load('Model/lr.sav')
rf_model = joblib.load('Model/rf.sav')
# Optional classifier models (preferred)
rf_clf = None
lr_clf = None
try:
    rf_clf = joblib.load('Model/rf_clf.sav')
except Exception:
    rf_clf = None
try:
    lr_clf = joblib.load('Model/lr_clf.sav')
except Exception:
    lr_clf = None

# Prefer an explicit saved list of feature columns if available
feature_columns = None
try:
    feature_columns = joblib.load('Model/feature_columns.pkl')
except Exception:
    feature_columns = None

# Optional additional encoders
gender_le = None
mlb_interest = None
try:
    gender_le = joblib.load('Model/gender_le.pkl')
except Exception:
    gender_le = None

try:
    mlb_interest = joblib.load('Model/mlb_interest.pkl')
except Exception:
    mlb_interest = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    model_used = None

    if request.method == 'POST':
        skills_input = request.form.get('skills')
        model_choice = request.form.get('model')

        skills = [skill.strip() for skill in skills_input.split(',')]
        # Collect and normalize inputs
        interests_input = request.form.get('interests', '')
        gender_input = request.form.get('gender', '')
        grades_input = request.form.get('grades', '')

        # Transform skills into binary skill columns using the fitted mlb
        try:
            skill_matrix = mlb.transform([skills])
        except NotFittedError:
            msg = ("Error: the saved `MultiLabelBinarizer` is not fitted."
                   " Recreate `Model/mlb.pkl` from the training notebook (save the fitted"
                   " skills binarizer), then restart the app.")
            return render_template('index.html', prediction=msg, model_used=None)

        # Determine expected columns order: prefer `feature_columns.pkl`, then scaler.feature_names_in_
        if feature_columns is not None:
            expected_cols = list(feature_columns)
        elif hasattr(scaler, 'feature_names_in_'):
            expected_cols = list(scaler.feature_names_in_)
        else:
            expected_cols = None

        if expected_cols is not None:
            # create a zeroed row
            row = pd.DataFrame([np.zeros(len(expected_cols))], columns=expected_cols)

            # map skill binary columns into the row where column names match mlb.classes_
            skill_cols = list(mlb.classes_)
            for i, col in enumerate(skill_cols):
                if col in row.columns:
                    row.at[0, col] = skill_matrix[0, i]

            # Map interests (if we have an interest mlb)
            if mlb_interest is not None and interests_input:
                interests = [s.strip().lower() for s in interests_input.replace(';', ',').split(',') if s.strip()]
                try:
                    interest_matrix = mlb_interest.transform([interests])
                    interest_cols = list(mlb_interest.classes_)
                    for i, col in enumerate(interest_cols):
                        if col in row.columns:
                            row.at[0, col] = interest_matrix[0, i]
                except Exception:
                    # ignore interest mapping failures
                    pass

            # Map gender if possible
            if 'gender' in row.columns and gender_input:
                # try numeric first
                try:
                    row.at[0, 'gender'] = float(gender_input)
                except Exception:
                    if gender_le is not None:
                        try:
                            row.at[0, 'gender'] = int(gender_le.transform([gender_input])[0])
                        except Exception:
                            # fallback: leave as 0
                            pass

            # Map grades if provided
            if 'grades' in row.columns and grades_input:
                try:
                    row.at[0, 'grades'] = float(grades_input)
                except Exception:
                    pass

            try:
                scaled_features = scaler.transform(row)
            except Exception as e:
                msg = f"Error transforming features: {e}. Check that the encoder and scaler match training."
                return render_template('index.html', prediction=msg, model_used=None)
        else:
            # Fallback: scaler doesn't have feature names; attempt to pad the mlb output
            try:
                # create zero vector of expected length and insert mlb features at the end
                n_expected = getattr(scaler, 'n_features_in_', None)
                if n_expected is None:
                    raise RuntimeError('Scaler does not expose expected feature count')
                zeros = np.zeros((1, n_expected))
                mlb_len = skill_matrix.shape[1]
                if mlb_len > n_expected:
                    raise RuntimeError('mlb produces more features than scaler expects')
                zeros[0, :mlb_len] = skill_matrix
                scaled_features = scaler.transform(zeros)
            except Exception as e:
                msg = f"Error preparing features for scaler: {e}." \
                      f" Recreate scaler or ensure scaler was fit with DataFrame to preserve column names."
                return render_template('index.html', prediction=msg, model_used=None)

        # Prefer classifier models if available; otherwise fall back to regressors
        if model_choice == 'lr':
            if lr_clf is not None:
                try:
                    pred_class = lr_clf.predict(scaled_features)[0]
                    prediction = le.inverse_transform([int(pred_class)])[0]
                    model_used = "Logistic Regression (clf)"
                except Exception as e:
                    return render_template('index.html', prediction=f"Prediction error (lr clf): {e}", model_used=None)
            else:
                pred = lr_model.predict(scaled_features)
                model_used = "Linear Regression"
                try:
                    raw_pred = float(pred[0])
                except Exception:
                    try:
                        raw_pred = float(pred)
                    except Exception as e:
                        return render_template('index.html', prediction=f"Prediction error: {e}", model_used=None)

                import numpy as _np
                n_classes = len(le.classes_)
                idx = int(_np.round(raw_pred))
                idx = max(0, min(idx, n_classes - 1))
                try:
                    prediction = le.inverse_transform([idx])[0]
                except Exception:
                    indices = _np.arange(n_classes)
                    nearest = int(_np.argmin(_np.abs(indices - raw_pred)))
                    prediction = le.inverse_transform([nearest])[0]
        else:
            if rf_clf is not None:
                try:
                    pred_class = rf_clf.predict(scaled_features)[0]
                    prediction = le.inverse_transform([int(pred_class)])[0]
                    model_used = "Random Forest (clf)"
                except Exception as e:
                    return render_template('index.html', prediction=f"Prediction error (rf clf): {e}", model_used=None)
            else:
                pred = rf_model.predict(scaled_features)
                model_used = "Random Forest"
                try:
                    raw_pred = float(pred[0])
                except Exception:
                    try:
                        raw_pred = float(pred)
                    except Exception as e:
                        return render_template('index.html', prediction=f"Prediction error: {e}", model_used=None)

                import numpy as _np
                n_classes = len(le.classes_)
                idx = int(_np.round(raw_pred))
                idx = max(0, min(idx, n_classes - 1))
                try:
                    prediction = le.inverse_transform([idx])[0]
                except Exception:
                    indices = _np.arange(n_classes)
                    nearest = int(_np.argmin(_np.abs(indices - raw_pred)))
                    prediction = le.inverse_transform([nearest])[0]

    return render_template('index.html', prediction=prediction, model_used=model_used)

if __name__ == '__main__':
    app.run(debug=True)