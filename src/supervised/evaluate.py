import os
import sys
import tempfile
import warnings
from pathlib import Path
import json
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Salvar cache do matplotlib no diretório temporário do sistema operacional (e não no repositório)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / ".matplotlib"))
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

# Garantir que a pasta do script esteja no sys.path para imports locais
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from train_models import get_all_models, RANDOM_STATE

BASE_PATH = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_PATH / "data" / "processed"
RESULTS_PATH = BASE_PATH / "results"
TABLES_PATH = RESULTS_PATH / "tables"
FIGURES_PATH = RESULTS_PATH / "figures"


def load_data():
    train_file = DATA_PATH / "train.csv"
    test_file = DATA_PATH / "test.csv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Arquivos de dados não encontrados em {DATA_PATH}. "
            "Execute 'python src/supervised/preprocess.py' primeiro."
        )

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train = train_df.drop(columns=["class"])
    y_train = train_df["class"]

    X_test = test_df.drop(columns=["class"])
    y_test = test_df["class"]

    return X_train, y_train, X_test, y_test


def evaluate_all_models():
    TABLES_PATH.mkdir(parents=True, exist_ok=True)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()
    models = get_all_models()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
    }

    cv_records = []
    test_records = []
    best_params_dict = {}
    fitted_models = {}
    predictions = {}
    probabilities = {}

    print("=" * 70)
    print("Iniciando busca de hiperparâmetros e validação cruzada (5-fold CV)...")
    print("=" * 70)

    for name, (pipeline, param_grid) in models.items():
        print(f"\n[+] Treinando e otimizando: {name}...")

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            refit="f1_macro",
            n_jobs=-1,
            return_train_score=False,
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        fitted_models[name] = best_model
        best_params_dict[name] = grid_search.best_params_

        best_idx = grid_search.best_index_
        cv_res = grid_search.cv_results_

        cv_record = {
            "Modelo": name,
            "Accuracy_CV_Mean": cv_res["mean_test_accuracy"][best_idx],
            "Accuracy_CV_Std": cv_res["std_test_accuracy"][best_idx],
            "Precision_Macro_CV_Mean": cv_res["mean_test_precision_macro"][best_idx],
            "Precision_Macro_CV_Std": cv_res["std_test_precision_macro"][best_idx],
            "Recall_Macro_CV_Mean": cv_res["mean_test_recall_macro"][best_idx],
            "Recall_Macro_CV_Std": cv_res["std_test_recall_macro"][best_idx],
            "F1_Macro_CV_Mean": cv_res["mean_test_f1_macro"][best_idx],
            "F1_Macro_CV_Std": cv_res["std_test_f1_macro"][best_idx],
            "ROC_AUC_CV_Mean": cv_res["mean_test_roc_auc"][best_idx],
            "ROC_AUC_CV_Std": cv_res["std_test_roc_auc"][best_idx],
        }
        cv_records.append(cv_record)

        # Avaliação no conjunto de Teste Final (não tocado durante tuning)
        y_pred = best_model.predict(X_test)
        predictions[name] = y_pred

        if hasattr(best_model, "predict_proba"):
            y_proba = best_model.predict_proba(X_test)[:, 1]
            roc_auc_val = roc_auc_score(y_test, y_proba)
            probabilities[name] = y_proba
        else:
            y_proba = None
            roc_auc_val = np.nan

        test_record = {
            "Modelo": name,
            "Accuracy_Test": accuracy_score(y_test, y_pred),
            "Precision_Macro_Test": precision_score(y_test, y_pred, average="macro", zero_division=0),
            "Recall_Macro_Test": recall_score(y_test, y_pred, average="macro", zero_division=0),
            "F1_Macro_Test": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "ROC_AUC_Test": roc_auc_val,
        }
        test_records.append(test_record)

        print(f"    Melhores hiperparâmetros: {grid_search.best_params_}")
        print(f"    F1-Macro (CV):   {cv_record['F1_Macro_CV_Mean']:.4f} (+/- {cv_record['F1_Macro_CV_Std']:.4f})")
        print(f"    F1-Macro (Teste):{test_record['F1_Macro_Test']:.4f}")
        print(f"    ROC-AUC (Teste): {test_record['ROC_AUC_Test']:.4f}")

    # salvar tabelas
    df_cv = pd.DataFrame(cv_records)
    df_test = pd.DataFrame(test_records)
    df_comparison = pd.merge(
        df_cv[["Modelo", "F1_Macro_CV_Mean", "ROC_AUC_CV_Mean", "Accuracy_CV_Mean"]],
        df_test[["Modelo", "F1_Macro_Test", "ROC_AUC_Test", "Accuracy_Test", "Precision_Macro_Test", "Recall_Macro_Test"]],
        on="Modelo",
    )

    df_cv.to_csv(TABLES_PATH / "metrics_cv.csv", index=False)
    df_test.to_csv(TABLES_PATH / "metrics_test.csv", index=False)
    df_comparison.to_csv(TABLES_PATH / "metrics_comparison.csv", index=False)

    with open(TABLES_PATH / "best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params_dict, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Tabelas salvas em /results/tables/:")
    print(" - metrics_cv.csv")
    print(" - metrics_test.csv")
    print(" - metrics_comparison.csv")
    print(" - best_hyperparameters.json")
    print("=" * 70)

    # gera matrizes de confusao
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, y_pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Bom (1)", "Mau (0)"])
        disp.plot(ax=axes[idx], cmap=plt.cm.Blues, colorbar=False)
        axes[idx].set_title(f"Matriz de Confusão: {name}", fontsize=11, fontweight="bold")

    fig.delaxes(axes[5])
    plt.tight_layout()
    fig.savefig(FIGURES_PATH / "confusion_matrices.png", dpi=300)
    plt.close(fig)

    # 2. gera curvas roc
    plt.figure(figsize=(9, 7))
    for name, y_proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_val:.3f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Aleatório (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos (FPR)", fontsize=11)
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)", fontsize=11)
    plt.title("Curvas ROC no Conjunto de Teste", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "roc_curves.png", dpi=300)
    plt.close()

    # 3. gera grafico de barras comparativo
    metrics_to_plot = ["F1_Macro_Test", "ROC_AUC_Test", "Accuracy_Test"]
    labels_to_plot = ["F1-Macro", "ROC-AUC", "Acurácia"]
    x = np.arange(len(df_test))
    width = 0.25

    plt.figure(figsize=(10, 6))
    for i, (metric, label) in enumerate(zip(metrics_to_plot, labels_to_plot)):
        plt.bar(x + (i - 1) * width, df_test[metric], width, label=label)

    plt.xlabel("Modelos", fontsize=12)
    plt.ylabel("Pontuação (Score)", fontsize=12)
    plt.title("Desempenho dos Modelos Supervisionados no Conjunto de Teste", fontsize=13, fontweight="bold")
    plt.xticks(x, df_test["Modelo"], fontsize=10)
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_PATH / "models_performance_test.png", dpi=300)
    plt.close()

    # 4. Gerar visualizacao e regras da arvore de decisao
    if "Decision Tree" in fitted_models:
        dt_pipeline = fitted_models["Decision Tree"]
        preprocessor = dt_pipeline.named_steps["preprocessor"]
        dt_clf = dt_pipeline.named_steps["classifier"]

        feature_names = [
            f.replace("num__", "").replace("cat__", "")
            for f in preprocessor.get_feature_names_out()
        ]

        plt.figure(figsize=(24, 12))
        plot_tree(
            dt_clf,
            feature_names=feature_names,
            class_names=["Mau Pagador (0)", "Bom Pagador (1)"],
            filled=True,
            rounded=True,
            max_depth=3,
            fontsize=9,
        )
        plt.title(
            f"Árvore de Decisão - Visualização dos Primeiros 3 Níveis (Critério: {dt_clf.criterion})",
            fontsize=15,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(FIGURES_PATH / "decision_tree.png", dpi=300)
        plt.close()

        rules = export_text(dt_clf, feature_names=feature_names, max_depth=4)
        with open(TABLES_PATH / "decision_tree_rules.txt", "w", encoding="utf-8") as f:
            f.write(rules)

    print("Gráficos salvos em /results/figures/:")
    print(" - confusion_matrices.png")
    print(" - roc_curves.png")
    print(" - models_performance_test.png")
    print(" - decision_tree.png")
    print("Regras da árvore salvas em /results/tables/decision_tree_rules.txt")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_all_models()

