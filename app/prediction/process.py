from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd


# Dividir os dados em treino e teste
def create_model_linearRegression(df_features: pd.DataFrame, target):
    X_train, X_test, y_train, y_test = train_test_split(df_features, target, test_size=0.2, random_state=42)

    # Treinar um modelo de regressão linear
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Fazer previsões e avaliar o modelo
    y_pred = model.predict(X_test)
    # print(y_pred)

    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error (MSE): {mse}')

    r2 = r2_score(y_test, y_pred)
    print(f'R² Score: {r2}')