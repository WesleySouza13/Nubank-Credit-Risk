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

A aparente “boa separação” dos modelos, observada na matriz de confusão, é ilusória. 
Considerando isso, seguem as métricas de avaliação dos modelos:

<img width="805" height="133" alt="image" src="https://github.com/user-attachments/assets/e7fe5d62-b38c-4df7-b72b-3538c330d145" />

Realizei um cross-validation para verificar se os resultados insatisfatórios poderiam ser causados pela divisão dos dados em folds, mas os resultados continuaram abaixo do esperado:

<img width="596" height="144" alt="image" src="https://github.com/user-attachments/assets/5668e1be-8f5b-4a2c-a081-2198af5a8536" />

Com isso, prossegui com uma análise de multicolinearidade para identificar linearidades e redundâncias no dataset.

**Mini dicionario das metricas**

Deixarei um mini dicionario das metricas apenas para didatica:
| Métrica          | Significado                        |
| ---------------- | ---------------------------------- |
| **logloss**      | Erro nas probabilidades            |
| **brier\_score** | Calibração das probabilidades      |
| **f1\_score**    | Equilíbrio entre precisão e recall |
| **recall**       | Acerto dos positivos               |
| **roc\_auc**     | Separação entre classes            |
| **ks**           | Distância entre TPR e FPR          |

Ks será a métrica discriminante nesse estudo

# Analise de Multicolinearidade - VIF 

Utilizando o algoritmo de Fator de Inflação da Variância (Variance Inflation Factor), disponível na biblioteca statsmodels pelo método statsmodels.stats.outliers_influence.variance_inflation_factor, obtive as seguintes métricas:

<img width="554" height="456" alt="image" src="https://github.com/user-attachments/assets/c7d76a48-85fd-4b83-af36-e70089fdf1fd" />

Onde temos os seguintes intervalos: 

 valores entre 0 e 5 indicam baixa multicolinearidade
 
 entre 6 e 10 sugerem presença de multicolinearidade aceitável
 
 entre 10 e 13 representam um limite pessoal de multicolinearidade
 
 valores acima de 14 indicam problema significativo de multicolinearidade

 Com isso, temos a variável reported_income como um grande indicador de multicolinearidade.

Modelos baseados em árvores não sofrem com problemas de multicolinearidade, porém estamos trabalhando também com um modelo linear (regressão logística), que é bastante afetado por esse problema, impactando seu desempenho.

Um ponto que gosto de destacar é que a multicolinearidade implica dispersão e redundância de informações nos dados, ou seja, teremos vetores informando essencialmente a mesma coisa para o modelo. Com isso em mente, surge a dúvida: esse problema não deveria indicar um desempenho artificialmente alto para o modelo? Por que ele está apresentando underfitting? Esse foi o meu primeiro questionamento sobre meu target.

Seguimos...

# Criaçao da Classe FeatureEng

Para enriquecer os modelos com mais informações, decidi criar uma classe que realiza esse processamento sem causar Data Leakage (vazamento de dados). Assim surgiu a instância FeatureEng, que recebe os valores x (dados observados) e os transforma em novas informações.

Observação: a criação dessa nova classe também foi pensada para encapsulamento em uma pipeline, permitindo que os dados já sejam transformados diretamente no ambiente de produção.

Paralelamente, criei a classe Binarize.py, responsável por criar bins de renda. A classe FeatureEng herda essa funcionalidade, garantindo que a transformação de dados seja organizada e reutilizável.

**FeatureEng**

<img width="1305" height="868" alt="image" src="https://github.com/user-attachments/assets/992ae8cc-cb1f-4d83-975b-309387762618" />

**Binarize**

<img width="630" height="698" alt="image" src="https://github.com/user-attachments/assets/8e89887d-e3ff-405e-a9fd-1970ce4752c9" />

Inclusive, alguns bins na classe FeatureEng foram desenvolvidos utilizando a biblioteca OptimalBinning. Ao longo desta apresentação, explicarei um pouco mais sobre esse processo

# SHAP Values

Para entender a influencia das variaveis em nossos modelos, realizei um estudo utilizando o SHAP. 

<img width="813" height="700" alt="image" src="https://github.com/user-attachments/assets/0094220b-3408-44fa-85ce-378ff9f3e1da" />

O resultado mostra que a variável renda não é uma boa feature para nossos modelos. Isso se deve pela influência sazonal da variável.

Vou explicar rapidamente:

Em nosso país, quem trabalha de carteira assinada (CLT) tem direito a 1 salário a mais na conta em uma época específica do ano, isso se chama 13º salário. Vamos supor que nosso amigo Joãozinho recebe R$ 2.500,00 no mês. Significa que a nossa classe Binarize vai classificar ele como renda média (2), certo? Mas, no mês de novembro, Joãozinho recebe 1 salário a mais em sua conta, isso quer dizer que vai ter o salário de R$ 2.500 + o 13º salário, totalizando R$ 5.000. E, nesse bendito mês, ele decide fazer uma consulta para conseguir um empréstimo, por exemplo. Se o modelo aprende que a renda é um discriminante, ele não vai avaliar todo o comportamento do Joãozinho antes da concessão, mas sim o que ele apresenta na hora da concessão. Isso cria um problema de falso comportamento, fazendo com que instituições financeiras não avaliem o cliente diretamente pela sua renda, pois é um indicador sazonal.

Modelos de Behavior Score se saem (e são indicados) muito bem nesse tipo de problema.


# OptimalBinning

Realizei um estudo utilizando o OptimalBinning e esses foram os resultados: 

<img width="896" height="462" alt="image" src="https://github.com/user-attachments/assets/0d18b200-382b-494b-86b8-17fd4f9763f6" />

<img width="651" height="552" alt="image" src="https://github.com/user-attachments/assets/7903abc0-6ff9-424f-bcfc-cf2c130b6773" />

<img width="648" height="552" alt="image" src="https://github.com/user-attachments/assets/9b2d2586-ba48-4e01-93e5-44ca1f7a5378" />


Analisando a coluna income, vemos que ela impacta negativamente a capacidade do modelo em conceder crédito. Isso se deve à característica sazonal da variável, ou seja, ela não reflete com precisão a realidade.

Se eu tiver uma renda baixa hoje, meu crédito poderá ser negado, independentemente do meu comportamento financeiro. Porém, se amanhã eu receber R$ 1.000 a mais do que o habitual, o modelo poderá aprovar meu crédito.

As estatísticas observadas são as seguintes:

Apenas 0,053545 ou 5% da carteira de clientes teria peso na renda para influenciar a decisão do modelo (WoE: 0,422522).

18% da carteira influencia negativamente a decisão do modelo, com importância negativa de -0,078047, mesmo mantendo comportamento conservador e vários meses em ok_since.


# Problema de Target 

Ao analisar a base de clientes inadimplentes, percebi que o comportamento dos mesmos não remete a um possível calote. Por exemplo, há clientes com mais de 32 meses em "ok" (sem boletos atrasados), indicador de risco baixo, boa renda (apesar de não ser uma variável tão significativa), sem falências etc., sendo classificados como 1 (inadimplentes). Ao contrário, clientes que estavam com falências, dívidas em aberto, poucos ou nenhum mês em "ok" foram classificados como 0 (adimplentes). 

# Modelando novo target 

Para criar um novo Y, não tratei isso de qualquer maneira. Primeiro, selecionei variáveis que são bons indicadores de risco, como número de empréstimos inadimplentes, número de problemas, falências, taxa de risco, consultas externas no último mês e ano, além do score_rating_enc.

Com essas variáveis, treinei um modelo de árvore de decisão com profundidade máxima igual a 3, balanceando as classes para evitar viés. Esse modelo serviu como uma base para gerar um score de risco, onde a saída foi a probabilidade prevista de inadimplência para cada cliente.

A partir desse score, calculei métricas de avaliação como AUC, log loss e também a curva ROC para obter o KS (Kolmogorov-Smirnov). O melhor ponto de corte foi definido pelo threshold que maximiza a separação entre bons e maus pagadores.

Com esse corte, gerei um novo target chamado y_target. Esse novo Y representa um grupo de clientes classificados como de alto risco (1) ou baixo risco (0), com base na probabilidade prevista pelo modelo e no ponto de corte calculado.

Por fim, validei a separação observando a taxa real de inadimplência em cada grupo do novo target, confirmando que o modelo conseguiu criar uma nova variável resposta que reflete de forma mais consistente o risco de crédito.

<img width="613" height="139" alt="image" src="https://github.com/user-attachments/assets/5632893c-b160-44ce-a1e1-ebcafcbb416f" />

# Nova rodada de treinamento 

Após novas rodadas de treinamento, decidi implementar o modelo de árvore de decisão.

Defendo minha escolha com os seguintes argumentos: como havia dados com multicolinearidade, escolhi não explorar a regressão logística, pois aplicar um PCA dificultaria a capacidade de explicabilidade do modelo, ainda mais em um problema onde entender o que está acontecendo é importante. Eu poderia excluir as variáveis redundantes para seguir com a regressão logística, mas isso faria o modelo perder informações relevantes.

Por outro lado, aplicar o XGBoost poderia parecer uma ótima escolha. Porém, suas métricas excessivamente altas levantam dúvidas sobre a real capacidade de separação do modelo.

Por fim, optei pelas árvores de decisão, pela sua capacidade de poda e pela clareza de interpretação. As métricas obtidas foram as seguintes:

{'model_name': 'DecisionTreeClassifier', 'logloss': 2.220446049250313e-16, 'brier_score': 0.0, 'f1_score': 1.0, 'recall': 1.0, 'roc_auc': 1.0, 'ks': 1.0}

# Optuna

Eu apliquei a busca de paramentros utilizando o optuna em todos os modelos, mas darei prioridade em explicar o modelo onde darei sequencia com deploy.
Para saber mais sobre os hiperparamentros dos demais modelos, acesse a pasta "optuna". 

# Hiperparamentros - Árvores de decisão

os parametros utilizados para a arvore de decisao pós optuna foram os seguintes:

max_depth = 12 → limita a profundidade da árvore para evitar overfitting e manter o modelo explicável.

min_samples_split = 11 → garante que só sejam feitos splits quando houver quantidade razoável de amostras, evitando divisões inúteis.

min_samples_leaf = 6 → força cada folha a ter pelo menos 6 registros, o que suaviza a árvore e reduz ruído.

max_features = None → deixa o modelo considerar todas as variáveis em cada split, aproveitando ao máximo a informação disponível.

criterion = 'gini' → mede impureza de forma eficiente, sendo padrão e estável para classificação binária.

random_state = 42 → garante reprodutibilidade dos resultados.

class_weight = 'balanced' → corrige o desbalanceamento entre classes, ajustando os pesos automaticamente.

Segue as métricas do modelo: 

recall: 1.0

brier_score: 0.07312192135554778

AUC: 0.9136507165411707

ks: 0.8273014330823415

# Separação de grupos homogeneos

Para separar clientes pelos seus comportamentos, decidi guiar uma analise de cluster utilizando Kmeans. Onde selecionei variaveis para separar os clientes em renda, faixa de score e limite de crédito. 

Primeiramente, realizei testes para calcular o numero de clusters ideais nessa divisao hoogenea. 

# Teste do cotovelo 

Os resultados do teste do cotovelo foram os seguintes:

<img width="813" height="682" alt="image" src="https://github.com/user-attachments/assets/8bbfce57-2f03-4026-9958-afc7bad0cbc7" />

O gráfico indica que o erro dos clusters apresenta uma queda mais acentuada a partir de k = 2.

# Silhouette Score e Davies-Bouldin Score

Com o objetivo de validar a divisão de clusters obtida pelo método do cotovelo, foram realizados testes de Silhouette Score e Davies-Bouldin Score.

Os resultados foram os seguintes:

<img width="972" height="208" alt="image" src="https://github.com/user-attachments/assets/90dd6e8e-4f41-4f67-83ee-a5bac2419e5c" />

No caso do Silhouette Score, valores mais próximos de 1 indicam melhor separação dos clusters. Com isso, a melhor divisão se manteve em k = 2.

Já no Davies-Bouldin Score, valores mais próximos de 0 representam melhor qualidade dos clusters. Diferente do Silhouette, não há um intervalo exato definido (como 0–1 ou 0–10), apenas a regra de que quanto menor, melhor.







