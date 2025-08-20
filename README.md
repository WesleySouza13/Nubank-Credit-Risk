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

target_default, score_3, risk_rate, credit_limit, income, n_defaulted_loans, n_accounts, n_issues, ok_since, n_bankruptcies, external_data_provider_credit_checks_last_year, external_data_provider_credit_checks_last_month, external_data_provider_credit_checks_last_2_year e reported_income. 

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


# Tratamento de Dados

Antes de tratar qualquer valor nulo utilizando métodos de imputação, busquei identificar as colunas que possuíam o maior volume de valores fora da curva. Com esse raciocínio, obtive as seguintes colunas: ['n_accounts', 'income', 'score_3'], nas quais o tratamento de outliers não impactaria negativamente a distribuição das informações da base. Para tratar esses valores fora da curva, utilizei a técnica de Winsorização, que consiste em identificar graficamente (por meio de boxplots) os limites superiores e inferiores dos dados e, em seguida, delimitar uma porcentagem para que os valores não ultrapassem esses limites. 

# Winsorização

A Winsorização é uma técnica de limites estatísticos utilizada para tratar a distribuição de valores fora da curva, analisando os limites inferiores e superiores da variável em estudo. O uso da Winsorização foi justificado pela necessidade de substituir possíveis valores nulos pela média ou pela mediana, já que a presença de outliers poderia exercer forte influência e distorcer essas medidas de tendência central. 


Resultados Winsorização: 

<img width="702" height="177" alt="image" src="https://github.com/user-attachments/assets/bf4c53de-7bc3-4136-b5ad-f3591f647d56" />

Após o tratamento dos outliers, os valores nulos das colunas ajustadas foram substituídos pela mediana, de modo a representar um comportamento central do cliente sem introduzir vieses adicionais na distribuição dos dados.

# Tratamento de target_default

A variável-alvo estava representada como valores booleanos, distribuída em True (1) e False (0). Para transformá-la em valores numéricos discretos, utilizei o método .replace() do pandas. 
A lógica aplicada foi a seguinte:

df2['target_default'] = df2['target_default'].replace(True, 1)

df2['target_default'] = df2['target_default'].replace(False, 0)

Veja graficamente o resultado da distribuição do target:

<img width="1003" height="682" alt="image" src="https://github.com/user-attachments/assets/d935f0c3-35ae-48bb-8ba8-403a05f2dec1" />

Ja indicando a presença de desbalanceamento das classes, comportamento comum e esperado em problemas de risco de crédito. 

# Tratamento de Valores infinitos 
A coluna reported_income representa a característica da base de dados que indica se o cliente possui renda declarada. Ela deveria se comportar como uma variável contínua ou categórica binária (0 ou 1). No entanto, a presença de valores np.inf alterou completamente o seu comportamento.

Para tratar esse problema, apliquei a seguinte lógica:

df2['reported_income'] = df2['reported_income'].replace([np.inf, -np.inf], np.nan)

Dessa forma, todos os valores infinitos foram substituídos por valores nulos. O total de valores nulos nessa variável foi de 66 registros, o que não foi estatisticamente significativo para impactar a análise, de modo que a exclusão desses casos não trouxe prejuízo relevante à base.

# O problema das variaveis: 'external_data_provider_credit_checks_last_year', 'external_data_provider_credit_checks_last_2_year'







