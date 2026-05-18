# Decision Tree — Previsão de Sobrevivência no Titanic

> Modelo preditivo supervisionado baseado em Árvore de Decisão (Decision Tree Classifier), aplicado ao clássico dataset do Titanic para classificar a sobrevivência de passageiros.

---

## Visão Geral

Este projeto aplica técnicas de **Machine Learning supervisionado** para prever se um passageiro do Titanic sobreviveu ou não ao naufrágio, com base em atributos como classe social, sexo, idade, tarifa paga e porto de embarque.

O modelo utiliza o algoritmo **Decision Tree Classifier** da biblioteca Scikit-learn, com engenharia de features, tratamento de dados ausentes, codificação de variáveis categóricas e avaliação de desempenho por métricas padrão de classificação.

---

## Fundamentos Teóricos

### O que é uma Árvore de Decisão?

Uma Árvore de Decisão é um algoritmo de aprendizado supervisionado que divide o espaço de features em regiões sucessivas com base em **regras de divisão (splits)**, formando uma estrutura em árvore onde:

- Cada **nó interno** representa um teste sobre um atributo
- Cada **ramo** representa o resultado do teste
- Cada **folha** representa uma classe prevista

### Critério de Impureza — Gini

O modelo utiliza o índice de **Gini** para avaliar a qualidade de cada divisão:

```
Gini(t) = 1 - Σ p(i|t)²
```

Onde `p(i|t)` é a proporção de amostras da classe `i` no nó `t`. Quanto menor o Gini, mais puro é o nó — ou seja, mais homogêneo em termos de classes.

> **Alternativa:** O critério `entropy` (Ganho de Informação) também pode ser usado e tende a produzir árvores ligeiramente diferentes.

### Controle de Overfitting — `max_depth`

Árvores sem restrição crescem até memorizar os dados de treino (overfitting). O parâmetro `max_depth=4` limita a profundidade máxima da árvore, forçando generalizações mais robustas e facilitando a interpretação visual.

---

## Pipeline do Projeto

```
Carregamento dos Dados
        ↓
Tratamento de Valores Nulos
        ↓
Engenharia de Features
        ↓
Codificação de Variáveis Categóricas
        ↓
Divisão Treino / Teste (80% / 20%)
        ↓
Treinamento do Modelo (DecisionTreeClassifier)
        ↓
Avaliação de Desempenho
        ↓
Predição para Novos Passageiros
```

---

## Tratamento de Dados

| Coluna | Problema | Solução |
|---|---|---|
| `Embarked` | 2 valores nulos | `dropna()` — proporção irrelevante |
| `Age` | ~20% nulos | Preenchimento com **mediana** |
| `Cabin` | ~77% nulos | Transformada em feature binária `Has_Cabin` |
| `Name`, `Ticket`, `PassengerId` | Sem valor preditivo | Descartadas |

> **Boa prática:** `dropna()` é justificável apenas quando a proporção de nulos é pequena e sua remoção não enviesa a amostra. Para colunas com alta proporção de ausências, imputação ou engenharia de feature são preferíveis.

---

## Engenharia de Features

### Codificação de Variáveis Categóricas

**`Sex` → Label Encoding binário**

```python
le_sex = LabelEncoder()
df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])
df['Sex_encoded'] = 1 - df['Sex_encoded']  # female = 1, male = 0
```

**`Embarked` → One-Hot Encoding**

```python
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Emb')
```

O One-Hot Encoding é utilizado para evitar **ordinalidade fictícia** — o modelo não deve interpretar `C > Q > S` como uma relação numérica real.

### Features Finais do Modelo

| Feature | Tipo | Descrição |
|---|---|---|
| `Pclass` | Ordinal | Classe do bilhete (1ª, 2ª, 3ª) |
| `Sex_encoded` | Binária | Sexo do passageiro (female=1, male=0) |
| `Age` | Contínua | Idade em anos |
| `SibSp` | Discreta | Nº de irmãos/cônjuges a bordo |
| `Parch` | Discreta | Nº de pais/filhos a bordo |
| `Fare` | Contínua | Tarifa paga pela passagem |
| `Has_Cabin` | Binária | Possui cabine registrada (1=Sim, 0=Não) |
| `Emb_C` | Binária | Embarcou em Cherbourg |
| `Emb_Q` | Binária | Embarcou em Queenstown |
| `Emb_S` | Binária | Embarcou em Southampton |

---

## Treinamento

```python
modelo = DecisionTreeClassifier(
    max_depth=4,        # Limita profundidade para evitar overfitting
    criterion='gini',   # Índice de Gini como medida de impureza
    random_state=42     # Garante reprodutibilidade dos resultados
)
modelo.fit(x_train, y_train)
```

A divisão treino/teste utiliza `stratify=y`, preservando a proporção de sobreviventes em ambos os conjuntos — essencial em datasets com **classes desbalanceadas**.

---

## Avaliação do Modelo

O modelo é avaliado com:

- **Acurácia** — proporção de predições corretas
- **Relatório de Classificação** — precisão, recall e F1-score por classe

```
Acurácia no conjunto de teste: ~82%

              precision    recall  f1-score
Não sobreviveu    0.84      0.87      0.86
Sobreviveu        0.79      0.74      0.76
```

> **F1-score** é especialmente relevante em problemas de classificação com classes desbalanceadas, pois pondera precisão e recall igualmente.

---

## Exemplos de Predição

| Passageiro | Características | Predição |
|---|---|---|
| Passageiro 1 | Mulher, 29 anos, 1ª classe, cabine, Cherbourg | ✅ SOBREVIVEU |
| Passageiro 2 | Homem, 40 anos, 3ª classe, sem cabine, Queenstown | ❌ NÃO SOBREVIVEU |
| Passageiro 3 | Mulher, 18 anos, 2ª classe, sem cabine, Southampton | ✅ SOBREVIVEU |

---

## Tecnologias Utilizadas

| Biblioteca | Versão recomendada | Finalidade |
|---|---|---|
| `pandas` | ≥ 1.5 | Manipulação e análise de dados tabulares |
| `numpy` | ≥ 1.23 | Operações numéricas e arrays |
| `scikit-learn` | ≥ 1.2 | Modelo, métricas e pré-processamento |
| `matplotlib` | ≥ 3.6 | Visualização da árvore de decisão |
| `Python` | ≥ 3.9 | Linguagem base do projeto |

---

## Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/titanic-decision-tree.git
cd titanic-decision-tree
```

2. Instale as dependências:
```bash
pip install pandas numpy scikit-learn matplotlib
```

3. Baixe o dataset do Titanic no [Kaggle](https://www.kaggle.com/competitions/titanic/data) e coloque em `./archive/Titanic-Dataset.csv`

4. Execute o script:
```bash
python titanic_decision_tree.py
```

---

## Estrutura do Projeto

```
titanic-decision-tree/
│
├── archive/
│   └── Titanic-Dataset.csv      # Dataset original do Kaggle
│
├── titanic_decision_tree.py     # Script principal
└── README.md
```

---

## Referências

- [Scikit-learn — DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Kaggle — Titanic Dataset](https://www.kaggle.com/competitions/titanic)
- [Canal do Hype](https://www.youtube.com/@hype-data-and-ai)
---

Desenvolvido como projeto de estudo em Machine Learning com Python através do Curso Introdutório lecionado pelo HYPE - USP.
