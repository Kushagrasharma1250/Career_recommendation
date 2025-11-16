import joblib
import os

def main():
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Model')
    sc_path = os.path.join(model_dir, 'sc.sav')
    out_path = os.path.join(model_dir, 'feature_columns.pkl')

    if not os.path.exists(sc_path):
        print(f"Scaler not found at {sc_path}.")
        return

    sc = joblib.load(sc_path)
    if hasattr(sc, 'feature_names_in_'):
        cols = list(sc.feature_names_in_)
    elif hasattr(sc, 'mean_'):
        cols = [f'col_{i}' for i in range(len(sc.mean_))]
    else:
        raise RuntimeError('Scaler does not expose feature names or mean_')

    joblib.dump(cols, out_path)
    print(f'Saved {len(cols)} feature column names to {out_path}')

if __name__ == '__main__':
    main()
