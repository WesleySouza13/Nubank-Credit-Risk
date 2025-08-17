<img width="1440" height="1444" alt="image" src="https://github.com/user-attachments/assets/5ec60f9a-f9c9-4357-b24c-4bf6cfc846ff" />


# **Projeto: Risco de Crédito - Nubank**

**Introduçao: 👋**

Este projeto tem como objetivo analisar e modelar uma solução para o problema de inadimplência, utilizando dados do Nubank. Foi desenvolvido por mim, Wesley Matos, durante meus estudos sobre modelagem de risco, com foco específico em risco de crédito.

**Contexto do Problema**:

A concessão de crédito é uma parte importante no contexto bancário, onde podemos encontrar ganhos, mas também corremos o risco de perdas.
O projeto foi baseado na modelagem de algoritmos que identificam um possível problema futuro ou uma possível perda para a instituição.
A solução se baseia na construção de um modelo de Application Score, onde utilizamos informações estáticas do solicitante no momento da análise de crédito.

**Estrutura do Projeto 🗂️**

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

