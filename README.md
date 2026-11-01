# Trabalho 2 - Inteligência Artificial: Aprendizado de Máquina e Computação Natural

**Instituição:** CEFET-MG - Campus Divinópolis  
**Departamento:** Departamento de Computação - Engenharia de Computação (2026/2)  
**Disciplina:** G05IART0.02 - Inteligência Artificial  
**Professor:** Dr. Tiago Alves de Oliveira

---

## Estrutura do Repositório

```text
trabalho-ia-2026/
├── data/
│   ├── raw/
│   │   └── german.data                  # Dados brutos originais do UCI
│   ├── processed/
│   │   ├── train.csv                    # Conjunto de treino (80% estratificado)
│   │   └── test.csv                     # Conjunto de teste final (20% estratificado)
│   └── README.md                        # Documentação detalhada da base e atributos
├── notebooks/
│   └── eda.ipynb                        # Análise Exploratória de Dados executada
├── results/
│   ├── figures/                         # Gráficos (ROC, matrizes de confusão, EDA, árvore)
│   └── tables/                          # Tabelas de métricas, parâmetros e regras
├── src/
│   ├── supervised/
│   │   ├── preprocess.py                # Separação e pipeline de pré-processamento
│   │   ├── train_models.py              # Definição dos 5 modelos e grids de hiperparâmetros
│   │   └── evaluate.py                  # Avaliação via 5-fold CV e avaliação no teste
│   └── optimization/                    # Parte 2: Meta-heurísticas (AG, Enxame, Imunes)
├── reports/
│   └── relatorio.pdf                    # Relatório técnico final (6 a 10 páginas)
├── run.sh                               # Script bash automatizado de execução
├── requirements.txt                     # Dependências do projeto
└── README.md
```

---

## Parte 1: Base de Dados e Aprendizado Supervisionado

- **Base de Dados:** *Statlog (German Credit Data)* - [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **Tarefa:** Classificação binária de concessão de crédito (1: Bom Pagador, 0: Mau Pagador).
- **Amostras:** 1.000 instâncias (700 Bons / 300 Maus — proporção 70/30).
- **Atributos:** 20 atributos (7 numéricos e 13 categóricos).
- **Protocolo de Validação:**
  - Particionamento estratificado de 80% treino (800 amostras) e 20% teste (200 amostras) com `random_state=42`.
  - Validação cruzada estratificada em 5 folds (`StratifiedKFold`) aplicada exclusivamente no treino para seleção de hiperparâmetros (`GridSearchCV`).
  - Prevenção rigorosa de *Data Leakage* via `Pipeline` e `ColumnTransformer` (`StandardScaler` para numéricas e `OneHotEncoder` para categóricas).
  - Modelos avaliados: **Árvore de Decisão**, **KNN**, **SVM**, **Random Forest** e **AdaBoost**.

---

## Instalação e Requisitos

Recomenda-se o uso do Python 3.10 ou superior em ambiente virtual (`venv`):

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv venv
# No Linux/macOS:
source venv/bin/activate
# No Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# 2. Instalar as dependências
pip install -r requirements.txt
```

---

## Como Executar a Parte 1

O projeto conta com o script unificado `run.sh`:

```bash
# Executar apenas o pré-processamento (divisão estratificada e salvamento)
./run.sh preprocess

# Executar a avaliação completa dos 5 modelos supervisionados
./run.sh evaluate

# Executar todo o pipeline da Parte 1
./run.sh all
```

Alternativamente, os scripts em Python podem ser acionados diretamente:

```bash
# Pré-processamento
python src/supervised/preprocess.py

# Treinamento, Otimização e Avaliação
python src/supervised/evaluate.py
```

As tabelas de métricas (`results/tables/`) e gráficos comparativos (`results/figures/`) serão regerados automaticamente.
