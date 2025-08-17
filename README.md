<img width="1024" height="600" alt="image" src="https://github.com/user-attachments/assets/e3c2f387-588d-4d89-91b6-751a24107221" />


# **Projeto: Risco de Crédito - Nubank**

# **Introduçao 👋**

Este projeto tem como objetivo analisar e modelar uma solução para o problema de inadimplência, utilizando dados do Nubank. Foi desenvolvido por mim, Wesley Matos, durante meus estudos sobre modelagem de risco, com foco específico em risco de crédito.

# **Contexto do Problema**

A concessão de crédito é uma parte importante no contexto bancário, onde podemos encontrar ganhos, mas também corremos o risco de perdas.
O projeto foi baseado na modelagem de algoritmos que identificam um possível problema futuro ou uma possível perda para a instituição.
A solução se baseia na construção de um modelo de Application Score, onde utilizamos informações estáticas do solicitante no momento da análise de crédito.

# **Estrutura do Projeto 🗂️**

├── 📁 app/

├── 📁 dados/

│ ├── 📄 aquisição_train.csv

│ ├── 📄 data_newTarget.csv

│ └── 📄 data_tratado.csv

├── 📁 modelos/

├── 📁 cadernos/

│ ├── 📝 EDA.ipynb

│ └── 📝 modelagem.ipynb

├── 📁 optuna/

│ ├── 📝 DecisionTreeOptuna.ipynb

│ ├── 📝 LogisticOptuna.ipynb

│ └── 📝 XGBoostOptuna.ipynb

├── ⚙️ binarizer.py

├── 🖼️ Decision_tree_viz.png

├── 📄 decision_tree.dot

├── ⚙️ FeatureEng.py

├── 📄 requirements.txt

├── 📄 .gitignore

└── 📄 .pre-commit-config.yaml

# Explicando a base de dados

Eu fiz uma limpeza inicial selecionando apenas as variáveis significativas para o problema de negócio.
Por exemplo, variáveis como: marketing_channel, profile_phone_number, shipping_zip_code, entre outras, foram descartadas.
Com isso, criei um filtro para selecionar apenas as seguintes colunas:

target_default, score_3, risk_rate, credit_limit, income, n_defaulted_loans, n_accounts, n_issues, ok_since, n_bankruptcies, external_data_provider_credit_checks_last_year, external_data_provider_credit_checks_last_month, external_data_provider_credit_checks_last_2_year.

Essas variáveis dizem respeito ao comportamento financeiro dos clientes.

**Dicionario de Dados 📖**

| Variável                                                    | Descrição                                                                       |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **target\_default**                                         | Variável alvo → indica se o cliente entrou em inadimplência (`1`) ou não (`0`). |
| **score\_3**                                                | Pontuação de crédito pré-existente do cliente (escala interna).                 |
| **risk\_rate**                                              | Taxa de risco associada ao cliente, calculada por fontes externas/internas.     |
| **credit\_limit**                                           | Limite de crédito disponível para o cliente.                                    |
| **income**                                                  | Renda declarada do cliente.                                                     |
| **n\_defaulted\_loans**                                     | Número de empréstimos anteriores em que o cliente entrou em default.            |
| **n\_accounts**                                             | Número de contas/relacionamentos ativos do cliente no banco.                    |
| **n\_issues**                                               | Número de ocorrências ou problemas registrados (ex: atraso em pagamentos).      |
| **ok\_since**                                               | Tempo (em meses) desde que o cliente mantém histórico sem inadimplência.        |
| **n\_bankruptcies**                                         | Quantidade de falências registradas no histórico do cliente.                    |
| **external\_data\_provider\_credit\_checks\_last\_year**    | Número de consultas ao crédito feitas por terceiros no último ano.              |
| **external\_data\_provider\_credit\_checks\_last\_month**   | Número de consultas ao crédito feitas por terceiros no último mês.              |
| **external\_data\_provider\_credit\_checks\_last\_2\_year** | Número de consultas ao crédito feitas por terceiros nos últimos 2 anos.         |


