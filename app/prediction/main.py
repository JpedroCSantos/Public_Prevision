import pandas as pd
# Leitura e pré-processamento dos dados
DATA_PATH = "data/output"
FINAL_PATH = f"{DATA_PATH}/DATAS/FINAL_DATABASE"
FINAL_FILE_NAME = "MOVIES"

TEMPORARY_FILE_NAME    = "TEMPORARY_FILE_DATABASE"
TEMPORARY_FILES_PATH    = f"{DATA_PATH}/TESTES"

df = pd.read_parquet(f"{FINAL_PATH}/{FINAL_FILE_NAME}.parquet")

# Visualizar informações gerais
print(df.info())
print(df.describe())
# # Visualizar os primeiros registros
# print(df.head())
# # Verificar valores únicos para colunas categóricas
# print(df.nunique())
# # Verificar valores ausentes por coluna
# print(df.isna().sum())
# # Visualizar as linhas com valores ausentes
# print(df[df.isna().any(axis=1)].head())
# df['IMDB_Rating'].fillna(df['IMDB_Rating'].mean(), inplace=True)
# print(df)


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from category_encoders import TargetEncoder
# from sklearn.preprocessing import StandardScaler, MinMaxScaler

# # Leitura e pré-processamento dos dados
# DATA_PATH = "data/output"
# FINAL_PATH = f"{DATA_PATH}/DATAS/FINAL_DATABASE"
# FINAL_FILE_NAME = "MOVIES"

# TEMPORARY_FILE_NAME    = "TEMPORARY_FILE_DATABASE"
# TEMPORARY_FILES_PATH    = f"{DATA_PATH}/TESTES"

# df = pd.read_parquet(f"{FINAL_PATH}/{FINAL_FILE_NAME}.parquet")

# # Transformações já aplicadas no seu código
# df = df.drop('Title', axis=1)
# df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
# df['Year'] = df['Release_Date'].dt.year
# df['Month'] = df['Release_Date'].dt.month
# df['Day_of_Week'] = df['Release_Date'].dt.dayofweek
# df['Day_of_Week_sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
# df['Day_of_Week_cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
# df = df.drop('Release_Date', axis=1)

# encoder_cast = TargetEncoder(cols=['Cast_1', 'Cast_2', 'Cast_3'])
# df[['Cast_1', 'Cast_2', 'Cast_3']] = encoder_cast.fit_transform(df[['Cast_1', 'Cast_2', 'Cast_3']], df['IMDB_Rating'])

# df = pd.get_dummies(df, columns=['Genre_1', 'Prodution_country'], prefix=['Genre', 'Country'])
# encoder = TargetEncoder(cols=['Production_Companies', 'Director_1'])
# df[['Production_Companies', 'Director_1']] = encoder.fit_transform(df[['Production_Companies', 'Director_1']], df['IMDB_Rating'])

# boolean_cols = df.select_dtypes(include='bool').columns
# df[boolean_cols] = df[boolean_cols].astype(int)
# df.to_csv(f"{TEMPORARY_FILES_PATH}/{TEMPORARY_FILE_NAME}.csv", index=False, sep = ",")

# numeric_cols = df.drop('Public_Total', axis=1).select_dtypes(include=['float64', 'int64']).columns
# scaler = MinMaxScaler()
# df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
# print(df)
# df.to_csv(f"{TEMPORARY_FILES_PATH}/{TEMPORARY_FILE_NAME}_SCALER.csv", index=False, sep = ",")

# # Separação de variável-alvo e features
# X = df.drop('Public_Total', axis=1)
# y = df['Public_Total']

# # Divisão em treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Treinamento do modelo de Regressão Linear
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Avaliação do modelo
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"MSE: {mse:.2f}")
# print(f"R²: {r2:.2f}")

# # Previsão com exemplo
# sample = X_test.iloc[0:1]
# predicted_public_total = model.predict(sample)
# print(f"Previsão para Public_Total: {predicted_public_total[0]:.2f}")

# correlation_matrix = df.corr()
# corr_with_target = correlation_matrix['Public_Total'].sort_values(ascending=False)
# print(corr_with_target)

# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(model, X, y, cv=5, scoring='r2')
# print("Cross-Validation R² Scores:", scores)
# print("Mean R²:", scores.mean())