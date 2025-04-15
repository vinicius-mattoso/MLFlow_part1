# 🍷 Wine Classifier — MLFlow_Part1

Este projeto é um exemplo simples de como integrar **MLflow** para rastreamento de experimentos em um modelo de classificação com o dataset clássico **Wine** do Scikit-Learn.

---

## ✅ Funcionalidades

- Treinamento de modelo `RandomForestClassifier`
- Rastreamento de experimentos com **MLflow Tracking**
- Utilização de `config.yaml` para definir:
  - Nome do experimento
  - Nome base do *run*
- Nome do run é automaticamente incrementado com data e hora para evitar conflitos
- Logging de parâmetros, métricas e artefatos do modelo

---

## 🗂 Estrutura do Projeto
MLFlow_part1/ 
│
├── README.md 
├── requirements.txt 
├── config.yaml 
├── train.py
├── train_optuna.py

## ⚙️ Como usar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/MLFlow_part1.git
cd MLFlow_part1/
```

### 2. Instale as dependências
Crie um ambiente virtual (opcional):

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

```bash
pip install -r ../requirements.txt
```

### 3. Execute o script
```bash
python train.py
or
python train_optuna.py
```

## 📊 Visualizando o MLflow UI

Execute o seguinte comando no terminal:
```bash
mlflow ui
```
Depois, acesse no navegador:

```bash
http://localhost:5000
```

## 🧠 Sobre o config.yaml
Este arquivo permite ao usuário definir parâmetros sem editar o código diretamente.
```bash
experiment:
  name: "wine_experiment"
  run_name: "baseline_model"
```