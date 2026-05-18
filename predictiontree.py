import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# CARREGAR O DATASET DO TITANIC (TREINO)
df = pd.read_csv('C:/Projects/Studies_in_Py/archive/Titanic-Dataset.csv')
print(df.head())

# TRATAMENTO DE VALORES NULOS

#1) Dropna() só deve ser utilizada se a proporção de nulos for demasiadamente reduzida ou 
# irrelevante para análise
df.dropna(subset=['Embarked'], inplace=True)

#2) Preenchendo com a mediana (numéricos)
df['Age'].fillna(df['Age'].median(), inplace=True)

#3) Transformar em uma coluna binário (Sim/Não)
df['Has_Cabin'] = df['Cabin'].notnull().astype(int)

df.drop('Cabin', axis = 1, inplace = True)
print(df.head())

# ================================================================
#3) ENGENHARIA DE FEATURES E SELEÇÃO DE VARIÁVEIS 
# ================================================================
# - Pclass (número ordinal, já pronta)
# - Sex (Categória(nominal): Male/Female -> Converter para binário) 
# - SibSp(Numérica discreta, já pronta)
# - Parch(Numérica discreta, já pronta)
# - Fare(Numérica discreta, já pronta)
# - Has_Cabin(Binária)
# - Embarked (Categórica: S, C, Q) -> Transformar para dummies(one-hot)

# Outras colunas como 'Name', 'Ticket', 'PassengerId' serão descartadas para o modelo

#4.1. Codificar Sex: male = 0, female = 1
le_sex = LabelEncoder()
df['Sex_encoded'] = le_sex.fit_transform(df['Sex'])

df['Sex_encoded'] = 1 - df['Sex_encoded']

#4.2. Codificar Embarked com One-Hot encoding (Evitar ordinalidade fictícia)
embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Emb')
df = pd.concat([df, embarked_dummies], axis=1)

#4.3 Selecionar as colunas finais para x = (features) e y = (target)
feature_cols = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare'
                , 'Has_Cabin','Emb_C', 'Emb_Q', 'Emb_S']
x = df[feature_cols]
y = df['Survived']



# ================================================================
#5) DIVIDIR EM CONJUNTOS DE TREINO E TESTE 
# ================================================================

#Utilizaremos 80% dos dados para treinar e 20% para testar
# random_state = 42 garante reprodutibilidade

x_train, x_Test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state = 42, stratify=y
)

print("\n" + "="*50)
print("Dimensões do Conjunto")
print('='*50)
print(f"Treino: {x_train.shape[0]} amostras")
print(f"Teste: {x_Test.shape[0]} amostras")


# ================================================================
#6) DIVIDIR EM CONJUNTOS DE TREINO E TESTE 
# ================================================================

#max_depth = 4: Limita a profundidade para evitar overlifting e facilitar a visualização.
#criterion='gini': medida de impureza(Alternative: 'entropy')
#random_state = 42: Reprodutibilidade

modelo = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state = 42)
modelo.fit(x_train, y_train)

print("\nModelo treinado com sucesso!")


# ================================================================
#7) AVLIAR OS MODELOS NOS DADOS DE TESTE 
# ================================================================

y_pred = modelo.predict(x_Test)

acuracia = accuracy_score(y_test, y_pred)

print("\n" + '='*50)
print(f"ACURÁCIA NO CONJUNTO DE TESTE:{acuracia:.2f} ({acuracia * 100:.1f}%)")
print('='*50)

print("\nRELATÓRIO DE RECLASSIFICAÇÃO: ")
print(classification_report(y_test, y_pred, target_names=['Não sobreviveu', 'Sobreviveu']))


plt.figure(figsize=(20,10))
plot_tree(modelo,
          feature_names=feature_cols,
          class_names=['Não sobreviveu', 'Sobreviveu'],
          filled = True,
          rounded = True,
          fontsize = 10,
          proportion = True)
plt.title("Árvore de decisão - Sobrevivência no Titanic(max_depth = 4)", fontsize=16)
plt.tight_layout()
plt.show()


# ================================================================
#8) FAZER PREDIÇÕES PARA UM NOVO PASSAGEIRO (EXEMPLO) 
# ================================================================
novo_passageiro = pd.DataFrame({
    'Pclass': [1, 3, 2],
    'Sex_encoded': [1, 0, 1],
    'Age': [29, 40, 18],
    'SibSp': [0, 1, 0],
    'Parch': [0, 2, 1],
    'Fare': [100.0, 20.5, 50.0],
    'Has_Cabin': [1, 0, 0],
    'Emb_C': [1, 0, 0],
    'Emb_Q': [0, 1, 0],
    'Emb_S': [0, 0, 1]
})
novo_passageiro == novo_passageiro[feature_cols]
predicao = modelo.predict(novo_passageiro)
probabilidades = modelo.predict_proba(novo_passageiro)

print("\n" + "="*50)
print("PREDIÇÃO PARA UM NOVO PASSAGEIRO:")
print("="*50)

print(f"Características: Mulher, 29 anos, 1º classe, tarifa $100, com cabine, embarcou em Cherboug")
print(f"Classe prevista: {'SOBREVIVEU'if predicao[0] == 1 else 'NÃO SOBREVIVEU'}")
print(f"Probabilidades: Não sobreviveu = {probabilidades[0][0]:.2f}, Sobreviveu = {probabilidades[0][1]:.2f}")
print("\n")
print(f"Características: Homem, 40 anos, 3º classe, tarifa $20.5, sem cabine, embarcou em Queenstown")
print(f"Classe prevista: {'SOBREVIVEU'if predicao[0] == 1 else 'NÃO SOBREVIVEU'}")
print(f"Probabilidades: Não sobreviveu = {probabilidades[0][0]:.2f}, Sobreviveu = {probabilidades[0][1]:.2f}")
print("\n")
print(f"Características: Mulher, 18 anos, 2º classe, tarifa $50, sem cabine, embarcou em Southampton")
print(f"Classe prevista: {'SOBREVIVEU'if predicao[0] == 1 else 'NÃO SOBREVIVEU'}")
print(f"Probabilidades: Não sobreviveu = {probabilidades[0][0]:.2f}, Sobreviveu = {probabilidades[0][1]:.2f}")