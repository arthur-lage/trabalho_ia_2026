# Trabalho 2 — Inteligência Artificial

**Instituição:** CEFET-MG — Campus Divinópolis<br/>
**Curso:** Engenharia de Computação — 2026/2<br/>
**Disciplina:** Inteligência Artificial
**Integrantes:** Arthur Batista Lage e Silva; Rafael Ferreira da Rocha

O projeto reúne a comparação de cinco classificadores na base Statlog German Credit e a implementação própria de três meta-heurísticas para o problema 40 da lista da disciplina.

## Estrutura

~~~text
data/
  raw/german.data
  processed/train.csv
  processed/test.csv
  README.md
notebooks/eda.ipynb
reports/
  build_report.py
  relatorio.pdf
results/
  figures/
  tables/
src/
  supervised/
  optimization/
tests/
run.sh
requirements.txt
~~~

## Parte 1 — Aprendizado supervisionado

### Base de dados

Foi usada a **Statlog (German Credit Data)**, obtida no [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data). A base tem 1.000 observações, 20 atributos preditivos (7 numéricos e 13 categóricos) e o alvo binário de risco de crédito. O arquivo data/README.md documenta a origem, autoria, licença e significado das variáveis.

### Protocolo

- Partição estratificada de 80% para treino e 20% para teste, com semente 42.
- Validação cruzada estratificada em cinco folds e busca de hiperparâmetros exclusivamente no treino.
- Pipeline e ColumnTransformer mantêm padronização numérica e one-hot dentro de cada fold, evitando vazamento.
- Modelos: Árvore de Decisão, KNN, SVM, Random Forest e AdaBoost.
- Seleção pelo F1 macro; resultados de acurácia, precisão macro, revocação macro e ROC-AUC também são registrados.

### Execução

No Windows, a partir da raiz do repositório:

~~~powershell
python src/supervised/preprocess.py
python src/supervised/evaluate.py
~~~

No Git Bash ou WSL, os equivalentes são ./run.sh preprocess, ./run.sh evaluate e ./run.sh all. O comando all executa a Parte 1 completa.

Para executar a EDA, abra notebooks/eda.ipynb a partir da raiz do repositório e execute as células em ordem. As tabelas e figuras da análise ficam em results/tables/ e results/figures/.

## Parte 2 — Problema 40: progressões de acordes

O problema escolhido é o **40 — Geração de escalas musicais ou progressões de acordes sob regras**. A variante implementada gera progressões diatônicas de oito posições. Cada acorde é representado pelo grau da escala, um inteiro de 0 a 6. Repetições são permitidas; a restrição adotada proíbe apenas repetições em posições adjacentes.

### Formulação

As três instâncias são Dó maior, Lá menor harmônica e Lá menor natural. Para cada instância, uma solução é x = (x₁, …, x₈), com xᵢ ∈ {0, …, 6}. O mapeamento do grau para a nota e a qualidade do acorde é feito pela escala da instância.

A função objetivo é de maximização:

~~~text
conformidade(x) = 0,85 × média das preferências de transição
                  + 0,15 × preferência de cadência do penúltimo acorde
diversidade(x)  = número de graus distintos em x / 7
f(x)            = 100 × (0,80 × conformidade(x) + 0,20 × diversidade(x))
~~~

As tabelas de preferência harmônica e cadência estão em src/optimization/problem.py. Transições não listadas recebem preferência baixa; a ACO usa as mesmas preferências como heurística. A função retorna o valor original, a viabilidade e a lista de violações. Violações não são somadas como penalidade ao objetivo.

Uma progressão viável:

| Código | 0 | 3 | 4 | 0 | 5 | 3 | 4 | 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Acorde em Dó maior | C | F | G | C | Am | F | G | C |

Restrições comuns aos três métodos: início e fim na tônica (grau 0), dominante (grau 4) no interior, pelo menos três graus distintos e nenhum acorde repetido em posições adjacentes. A representação por graus da escala assegura que os acordes pertençam à tonalidade.

### Algoritmos e parâmetros

Os três algoritmos foram implementados no projeto, sem bibliotecas que forneçam meta-heurísticas prontas.

- **AG:** população 40, torneio de tamanho 3, elitismo 2, cruzamento de um ponto com probabilidade 0,90 e mutação por gene com probabilidade 0,12.
- **ACO:** 24 formigas por atualização, depósito pelas quatro melhores, alpha=1, beta=2 e evaporação 0,20. A construção incorpora feromônio e preferência harmônica.
- **CLONALG:** população 30, seleção de até 10 anticorpos, fator de clonagem 6, hipermutação adaptativa 0,05 + 0,45 × (1 − afinidade) e quatro reposições aleatórias por rodada.

Os três usam a mesma função objetivo, restrições, instâncias e orçamento. O reparo preserva o comprimento e recupera as restrições escolhidas.

### Execução e validação

~~~powershell
python -m src.optimization.experiment
python -m src.optimization.experiment --validate-only
python -m unittest discover -s tests -v
~~~

No Git Bash/WSL, use ./run.sh optimize, ./run.sh validate-optimization e ./run.sh test. Cada método e instância tem dez execuções, sementes 42 a 51 e orçamento de 2.000 avaliações da função objetivo por execução. Uma busca aleatória viável, com o mesmo orçamento, serve como referência. O script registra melhor resultado, média, desvio-padrão, viabilidade, tempo, contagem de avaliações e convergência.

Os resultados da Parte 2 são salvos em results/tables/optimization_runs.csv, optimization_summary.csv, optimization_convergence.csv e optimization_metadata.json, além da figura results/figures/optimization_convergence.png.

## Relatório e reprodução

O relatório técnico está em reports/relatorio.pdf. Para recriá-lo depois de regenerar as tabelas e figuras:

~~~powershell
python reports/build_report.py
~~~

Instale as dependências listadas em requirements.txt em um ambiente Python 3.10 ou superior. O projeto usa caminhos relativos à raiz do repositório.

