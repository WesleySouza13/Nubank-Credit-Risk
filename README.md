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
| **reported_income**                                         | Se o cliente possui renda declarada (0 ou 1)                                    |

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
As colunas external_data_provider_credit_checks_last_year e external_data_provider_credit_checks_last_2_year representam o número de consultas que o cliente realizou no último ano e nos últimos dois anos, respectivamente. Isso evidencia que são variáveis bastante significativas para a análise do negócio. Por esse motivo, não seria adequado simplesmente excluí-las ou eliminar a grande quantidade de valores ausentes (mais de 50% da base).

Diante disso, decidi implementar um método de imputação diferente do que estávamos utilizando até então.

# Inputação por aprendizado supervisionado

Para tratar o problema de valores ausentes nas colunas mencionadas anteriormente, decidi implementar um algoritmo de Machine Learning para prever esses possíveis valores faltantes. O modelo escolhido foi o HistGradientBoostingClassifier, que lida de forma eficiente com valores ausentes presentes nos dados.

**Documentação do modelo**📖: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html

A lógica utilizada foi a seguinte:

<img width="910" height="478" alt="image" src="https://github.com/user-attachments/assets/12758db6-82ec-4aa5-996b-6136c71820ec" />

**A respeito da analise multivariada**

Meu foco neste case não foi desenvolver uma análise exploratória aprofundada além de compreender a distribuição dos dados e identificar alguns padrões relevantes, mas sim construir um modelo de application score robusto. Por esse motivo, não incluí os resultados da análise exploratória neste README.

No entanto, caso você queira visualizar o que foi feito no EDA, sinta-se à vontade para acessar o arquivo EDA.ipynb, localizado na pasta notebooks.

# Estudo de Causalidade

Neste projeto, ao contrário de trabalhos anteriores, busquei explorar um estudo de causalidade. Atualmente, estou aprofundando meus estudos nesse tema e aplicar os conceitos neste projeto é uma forma de praticar o que venho aprendendo. O objetivo não foi desenvolver um estudo causal extenso, mas compreender a influência de algumas variáveis sobre a variável resposta Y. 

Neste estudo, conduzi uma exploração de causalidade utilizando a função logit (regressão logística) como meu discriminante.

Dividi os elementos do estudo em: variáveis de tratamento (𝑥), variável resposta (𝑦)e variáveis de controle (𝑐𝑜𝑛𝑓).

Para orientar a análise, defini algumas questões específicas:

1) Pergunta causal: “Quais variáveis realmente impactam o risco de inadimplência de um cliente, controlando por outras variáveis relevantes?”

2) Questão específica: “Qual é o efeito das variáveis external_data_provider_credit_checks_last_year e external_data_provider_credit_checks_last_month sobre o alvo target_default?”

Para responder à primeira pergunta, realizei o seguinte estudo: Selecionei as variáveis significativas para o alvo y, sendo a variável de tratamento n_issues, a variável resposta target_default e as variáveis de controle score_3, credit_limit, n_defaulted_loans e ok_since. Em seguida, treinei a função logit para estimar o efeito dessas variáveis sobre o risco de inadimplência. 
Os resultados obtidos foram os seguintes:

<img width="877" height="453" alt="image" src="https://github.com/user-attachments/assets/75d90708-e2b6-4572-ac53-18784ef3ec8f" />

- A variável n_defaulted_loans apresentou coeficiente positivo, indicando que dívidas passadas podem aumentar o risco de inadimplência. No entanto, o p-valor maior que 0,05 mostra que esse efeito não é estatisticamente significativo, portanto não podemos afirmar com confiança que dívidas passadas sejam um fator causal decisivo para um futuro modelo.

- Variáveis como credit_limit e ok_since mostraram efeitos estatisticamente significativos, indicando que são características relevantes para a modelagem. O efeito negativo de credit_limit sugere que limites de crédito mais altos estão associados a menor risco de inadimplência, enquanto o efeito negativo de ok_since indica que clientes com maior tempo de relacionamento tendem a apresentar menor risco.

- A variável n_issues também se mostrou importante, com coeficiente de aproximadamente 1%, impactando positivamente o modelo. Isso significa que quanto maior o número de problemas, maior a probabilidade de inadimplência.

- Por fim, a variável score_3 não apresentou influência relevante, mostrando-se menos significativa do que esperado para o problema em questão.

Para responder à segunda pergunta, utilizei como variáveis de tratamento external_data_provider_credit_checks_last_year e external_data_provider_credit_checks_last_month, sem incluir variáveis de controle. 

Os resultados obtidos foram os seguintes:

<img width="1144" height="354" alt="image" src="https://github.com/user-attachments/assets/02b580a4-f85c-424d-94ff-6e8ff378a368" />

- Tanto external_data_provider_credit_checks_last_year quanto external_data_provider_credit_checks_last_month se mostraram variáveis significativas para explicar a inadimplência. Seus coeficientes negativos indicam que um maior número de consultas externas está associado a menor risco de inadimplência.


# Modelagem

Na parte de modelagem do projeto, selecionei três modelos comumente utilizados no cenário de Application Score: regressão logística, árvores de decisão e XGBoost. As métricas para avaliação dos modelos foram escolhidas considerando previamente a existência de desbalanceamento de classes. Com isso, os resultados da primeira rodada de treinamento, utilizando apenas os dados tratados, foram os seguintes:

<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/9f0378d9-8866-4492-a7ba-9cd0c1110767" />
<img width="1127" height="701" alt="image" src="https://github.com/user-attachments/assets/0976e5ff-bf40-4284-8f44-ecabe5027c33" />
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/4ccd238d-c0e4-46e5-814c-63c20178df62" />
<img width="1127" height="701" alt="image" src="https://github.com/user-attachments/assets/f293fbd4-f775-4fc9-a406-5860d49b388e" />
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/4cc6f445-7fad-4113-8bf9-bc6827c5857d" />
<img width="1127" height="701" alt="image" src="https://github.com/user-attachments/assets/ec253d96-c453-4f89-a594-5022e416e709" />














