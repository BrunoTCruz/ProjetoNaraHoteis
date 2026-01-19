import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Criando DataFrame com os dados
df = pd.DataFrame({'investimento_marketing': [10, 20, 30, 40, 50, 60],
                    'vendas': [100, 130, 160, 180, 210, 240]})


X = df[['investimento_marketing']] 
y = df['vendas']  

# Treinar o modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Estender os dados para previsão até R$ 80 mil
X_extendido = pd.DataFrame({'investimento_marketing': [70, 80]})
y_pred = modelo.predict(X_extendido)

# Visualizar
plt.figure(figsize=(8, 5))
plt.scatter(df['investimento_marketing'], df['vendas'], color='green', label='Vendas reais')
plt.plot(X_extendido['investimento_marketing'], y_pred, color='orange', label='Linha de regressão estendida')
plt.xlabel('Investimento em Marketing (mil R$)')
plt.ylabel('Vendas (mil unidades)')
plt.title('Regressão Linear com Previsão até R$ 80 mil')
plt.grid(True)
plt.show()
