import mlflow
import yaml
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from datetime import datetime

# ------------------------------
# Função para carregar o config
# ------------------------------
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ------------------------------
# Script principal de treino
# ------------------------------
def main():
    # Carrega as configurações
    config = load_config()
    experiment_name = config["experiment"]["name"]
    run_base_name = config["experiment"]["run_name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{run_base_name}_{timestamp}"
    
    # Define o experimento no MLflow
    mlflow.set_experiment(experiment_name)

    # Inicia um novo run
    with mlflow.start_run(run_name=run_name):
        # Carrega o dataset
        data = load_wine()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2, random_state=42
        )

        # Cria e treina o modelo
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Faz previsões e avalia
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # Log de parâmetros e métricas no MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", acc)

        # Opcional: salva o modelo
        mlflow.sklearn.log_model(model, "model")

        print(f"Run '{run_name}' finalizado com acurácia: {acc:.4f}")

if __name__ == "__main__":
    main()
