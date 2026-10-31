from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier 
from preprocess import get_preprocessor

RANDOM_STATE = 42

def get_decision_tree ():
    pipeline = Pipeline([("preprocessor", get_preprocessor()), ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    param_grid = {
        "classifier__max_depth": [3, 5, 7, 10, None],
        "classifier__criterion": ["gini", "entropy"],
        "classifier__min_samples_split": [2, 5, 10]
    }
    return pipeline, param_grid

def get_svm():
    pipeline = Pipeline([("preprocessor", get_preprocessor()), ("classifier", SVC(random_state=RANDOM_STATE, probability=True))])
    param_grid = {
        "classifier__kernel": ["linear", "rbf"],
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", "auto"]
    }
    return pipeline, param_grid

def get_random_forest():
    pipeline = Pipeline([("preprocessor", get_preprocessor()), ("classifier", RandomForestClassifier(random_state=RANDOM_STATE))])
    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_split": [2, 5]
    }
    return pipeline, param_grid

def get_knn():
    pipeline = Pipeline([("preprocessor", get_preprocessor()), ("classifier", KNeighborsClassifier())])
    param_grid = {
        "classifier__n_neighbors": [3, 5, 7, 11, 15],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"]
    }
    return pipeline, param_grid

def get_adaboost():
    pipeline = Pipeline([("preprocessor", get_preprocessor()), ("classifier", AdaBoostClassifier(random_state=RANDOM_STATE))])
    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.1, 0.5, 1.0]
    }
    return pipeline, param_grid

def get_all_models():
    return {
        "Decision Tree": get_decision_tree(),
        "KNN": get_knn(),
        "SVM": get_svm(),
        "Random Forest": get_random_forest(),
        "AdaBoost": get_adaboost(),
    }
    
if __name__ == "__main__":
    models = get_all_models()
    print("Modelos configurados com sucesso:")
    for name, (pipe, grid) in models.items():
        print(f" - {name}: {len(grid)} hiperparâmetros a explorar")