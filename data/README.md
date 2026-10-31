# Identificação e Documentação da Base de Dados

## Nome
Statlog (German Credit Data)

## URL / Origem
- **Repositório:** UCI Machine Learning Repository
- **Link:** [https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
- **DOI:** [10.24432/C5NC77](https://doi.org/10.24432/C5NC77)

## Autoria
- **Autor:** Prof. Dr. Hans Hofmann
- **Instituição:** Institut für Statistik und Ökonometrie, Universität Hamburg (Alemanha)
- **Ano:** 1994

## Licença e Condições de Uso
- **Licença:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Condições:** Permite compartilhamento e adaptação para fins acadêmicos e comerciais com a devida atribuição da fonte e citação do autor original.

## Descrição do Domínio
O conjunto de dados pertence ao domínio de **finanças e concessão de crédito bancário (*Credit Scoring* / Análise de Risco de Crédito)**. 

O objetivo do negócio bancário é avaliar o perfil de um cliente que solicita um empréstimo para determinar a probabilidade de inadimplência (*default*). O banco precisa decidir entre conceder o crédito (baixo risco) ou recusá-lo (alto risco), equilibrando o custo financeiro de conceder crédito a um cliente que não irá pagar (erro mais grave) versus o custo de oportunidade de recusar um bom pagador.

## Número de Instâncias
- **Total:** 1.000 instâncias (amostras reais de clientes).
- **Distribuição:**
  - **700 clientes classificados como Bons Pagadores (*Good Credit*)** (70%)
  - **300 clientes classificados como Maus Pagadores (*Bad Credit*)** (30%)
- **Desbalanceamento:** Apresenta desbalanceamento moderado de classes (razão 7:3), ideal para justificar o uso de métricas como F1-score macro, precisão macro, revocação macro e matriz de confusão.

## Atributos (Features)
A base é composta por **20 atributos preditivos** (7 numéricos e 13 categóricos/qualitativos) que descrevem a situação financeira, estabilidade profissional e perfil pessoal do cliente:

### Atributos Categóricos / Qualitativos (13)
1. **`checking_status` (Attribute 1):** Situação da conta corrente existente (`A11`: < 0 DM, `A12`: 0 <= x < 200 DM, `A13`: >= 200 DM / conta salário há pelo menos 1 ano, `A14`: sem conta corrente).
2. **`credit_history` (Attribute 3):** Histórico de crédito (`A30`: nenhum crédito tomado / todos pagos em dia, `A31`: todos os créditos pagos em dia neste banco, `A32`: créditos existentes em dia até o momento, `A33`: atrasos de pagamento no passado, `A34`: conta crítica / outros créditos em outros bancos).
3. **`purpose` (Attribute 4):** Finalidade do empréstimo (`A40`: carro novo, `A41`: carro usado, `A42`: móveis/equipamentos, `A43`: rádio/TV, `A44`: eletrodomésticos, `A45`: reparos, `A46`: educação, `A48`: capacitação/retreinamento, `A49`: negócios, `A410`: outros).
4. **`savings_status` (Attribute 6):** Poupança e títulos de investimento (`A61`: < 100 DM, `A62`: 100 <= x < 500 DM, `A63`: 500 <= x < 1000 DM, `A64`: >= 1000 DM, `A65`: desconhecido / sem poupança).
5. **`employment` (Attribute 7):** Tempo no emprego atual (`A71`: desempregado, `A72`: < 1 ano, `A73`: 1 <= x < 4 anos, `A74`: 4 <= x < 7 anos, `A75`: >= 7 anos).
6. **`personal_status` (Attribute 9):** Estado civil e sexo (`A91`: homem divorciado/separado, `A92`: mulher divorciada/separada/casada, `A93`: homem solteiro, `A94`: homem casado/viúvo, `A95`: mulher solteira).
7. **`other_parties` (Attribute 10):** Outros devedores / fiadores (`A101`: nenhum, `A102`: co-requerente, `A103`: fiador).
8. **`property_magnitude` (Attribute 12):** Bens patrimoniais (`A121`: imóvel próprio, `A122`: consórcio imobiliário / seguro de vida, `A123`: veículo próprio ou outros, `A124`: desconhecido / sem bens).
9. **`other_payment_plans` (Attribute 14):** Outros planos de financiamento em andamento (`A141`: outros bancos, `A142`: lojas comerciais, `A143`: nenhum).
10. **`housing` (Attribute 15):** Situação de moradia (`A151`: alugada, `A152`: própria, `A153`: gratuita / de favor).
11. **`job` (Attribute 17):** Tipo de ocupação (`A171`: desempregado / não qualificado não-residente, `A172`: não qualificado residente, `A173`: empregado qualificado / funcionário público, `A174`: gerência / autônomo / altamente qualificado).
12. **`own_telephone` (Attribute 19):** Possui telefone cadastrado no nome (`A191`: não, `A192`: sim).
13. **`foreign_worker` (Attribute 20):** Trabalhador estrangeiro (`A201`: sim, `A202`: não).

### Atributos Numéricos (7)
14. **`duration` (Attribute 2):** Duração do empréstimo em meses.
15. **`credit_amount` (Attribute 5):** Valor solicitado do crédito (em Marcos Alemães - DM).
16. **`installment_commitment` (Attribute 8):** Taxa das parcelas em porcentagem da renda disponível do cliente (1% a 4%).
17. **`residence_since` (Attribute 11):** Tempo de permanência na residência atual (em anos).
18. **`age` (Attribute 13):** Idade do proponente (em anos).
19. **`existing_credits` (Attribute 16):** Quantidade de empréstimos ativos existentes neste banco.
20. **`num_dependents` (Attribute 18):** Número de dependentes financeiros sob responsabilidade do cliente.

## Variável Alvo
- **Nome original:** `class` (ou `credit_risk`)
- **Valores originais:**
  - `1`: Bom pagador (*Good Credit*)
  - `2`: Mau pagador (*Bad Credit*)
- **Tratamento no pré-processamento:** Será mapeada para representação binária padrão:
  - `0`: Mau pagador / Inadimplente (*Bad*)
  - `1`: Bom pagador (*Good*)

## Tipo de Tarefa
**Aprendizado Supervisionado — Classificação Binária**.

## Ausência de Valores Faltantes
A base de dados original não apresenta valores nulos/ausentes registrados (`Has Missing Values: No`). Ainda assim, o pipeline de pré-processamento incorporará rotinas de validação e imputação preventiva (`SimpleImputer`) para assegurar robustez.