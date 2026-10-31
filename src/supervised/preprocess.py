from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_PATH = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = BASE_PATH / "data" / "raw" / "german.data"
PROCESSED_PATH = BASE_PATH / "data" / "processed"

COLUMN_NAMES = ["checking_status", "duration", "credit_history", "purpose", "amount", "savings_account", "employment_status", "installment_rate", "sex_and_marital_status", "other_debtors", "residence_since", "property", "age", "other_installments", "housing", "existing_credits_count_curr_bank", "job", "dependent_count", "telephone", "foreign_worker", "class"]
NUMERICAL_COLS = ["duration", "amount", "installment_rate", "residence_since", "age", "existing_credits_count_curr_bank", "dependent_count"]
CATEGORICAL_COLS = [col for col in COLUMN_NAMES if col not in NUMERICAL_COLS and col != 'class']

def load_raw_data ():
    df = pd.read_csv(RAW_DATA_PATH, sep=r"\s+", header=None, names=COLUMN_NAMES)

    # bom pagador vira 1, mau pagador vira 0
    df["class"] = df["class"].map({1:1, 2:0})

    return df

def get_preprocessor():
    return ColumnTransformer(transformers=[("num", StandardScaler(), NUMERICAL_COLS), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS)])

def split_and_save_data(test_size=0.2, random_state=42):
    df = load_raw_data()
    X = df.drop(columns=["class"])
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(PROCESSED_PATH / "train.csv", index=False)
    test_df.to_csv(PROCESSED_PATH / "test.csv", index=False)

    print("Dados divididos com sucesso!")
    print(f"Treino: {train_df.shape} | Teste: {test_df.shape}")
    print("==> Treino: /data/processed/train.csv")
    print("==> Teste: /data/processed/test.csv")



if __name__ == "__main__":
    split_and_save_data()