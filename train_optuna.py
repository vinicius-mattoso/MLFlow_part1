import mlflow
import optuna
import yaml
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# Função para carregar config
# ------------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ------------------------------
# Função de objetivo para o Optuna
# ------------------------------
def objective(trial):
    # Carrega os dados
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    # Sugestão de hiperparâmetros
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32)

    # Cria o modelo
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Treina e avalia
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Nome do run com timestamp e número do trial
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"optuna_trial_{trial.number}_{timestamp}"

    # Log no MLflow
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

    return acc

# ------------------------------
# Função principal
# ------------------------------
def main():
    # Carrega config
    config = load_config()
    experiment_name = config["experiment"]["name"]

    # Define o experimento
    mlflow.set_experiment(experiment_name)

    # Cria o estudo Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    print("Melhor trial:")
    print(study.best_trial)

if __name__ == "__main__":
    main()
