# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np

# # Separação em treino e teste
# X = df.drop(['Public_Total', 'Title'], axis=1)  # Excluir a variável-alvo e outras colunas irrelevantes
# y = df['Public_Total']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Aplicar Random Forest Regressor
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)

# # Avaliação do Modelo
# y_pred_rf = rf_model.predict(X_test)
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# r2_rf = r2_score(y_test, y_pred_rf)

# print(f"Random Forest Regressor - MSE: {mse_rf:.2f}")
# print(f"Random Forest Regressor - R²: {r2_rf:.2f}")

# # Cross-Validation
# cv_scores_rf = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
# print(f"Cross-Validation R² Scores (Random Forest): {cv_scores_rf}")
# print(f"Mean R² (Random Forest): {cv_scores_rf.mean():.2f}")
