import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import numpy as np

def random_forest_model(df: pd.DataFrame):
    """
    Função para treinar um modelo Random Forest, realizar EDA e avaliação completa do modelo.
    
    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados de entrada.
    
    Retorno:
    dict: Dicionário com os resultados do modelo (MSE, R², Importância das Variáveis, etc.).
    """
    df_cleaned = df.dropna()
    Q1 = df_cleaned.quantile(0.25)
    Q3 = df_cleaned.quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df_cleaned[~((df_cleaned < (Q1 - 1.5 * IQR)) | (df_cleaned > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    df_cleaned['Log_Public_Total'] = np.log1p(df_cleaned['Public_Total'])
    df_cleaned['Log_Days_in_exibithion'] = np.log1p(df_cleaned['Days_in_exibithion'])

    encoder = ce.TargetEncoder(cols=['Prodution_country', 'Genre_1', 'Production_Companies', 'Cast_1', 'Director_1'])
    X_encoded = encoder.fit_transform(df_cleaned.drop(columns=['Public_Total']), df_cleaned['Public_Total'])
    
    scaler = StandardScaler()
    X_encoded[['Days_in_exibithion', 'Vote_Average', 'Runtime', 'IMDB_Rating']] = scaler.fit_transform(
        X_encoded[['Days_in_exibithion', 'Vote_Average', 'Runtime', 'IMDB_Rating']]
    )
    
    X = X_encoded
    y = df_cleaned['Public_Total']  # Variável dependente
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='neg_mean_squared_error')

    importances = rf_model.feature_importances_
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

    print(f"\n📊 Resultados do Random Forest:")
    print(f"🔹 Mean Squared Error (MSE): {mse:.2f}")
    print(f"🔹 R²: {r2:.2f}")
    print(f"🔹 'Cross-validation MSE': {-cv_scores.mean()}")
    print(f"🔹 'Feature Importance': {feature_importance.sort_values(by='Importance', ascending=False)}")

    # 14. Ajuste de Hiperparâmetros com GridSearchCV (opcional)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    print(f"🔹 'Best Parameters': {grid_search.best_params_}")