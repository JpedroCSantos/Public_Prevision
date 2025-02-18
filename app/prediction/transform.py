import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import Dict, List

def build_target_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df_features = df.drop(target, axis=1) 

    return df_features

def enconding_features(features, df_features: pd.DataFrame) -> pd.DataFrame:
    if isinstance(features, list):
        concatenated_columns = pd.DataFrame()
        for feature in features:
            matching_columns = df_features.filter(like=feature)
            concatenated_columns = pd.concat([concatenated_columns, matching_columns])
        feature_to_encode = concatenated_columns
    else:
        feature_to_encode = df_features[features]
    
    all_unique_values = pd.Series(feature_to_encode.values.ravel()).unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(all_unique_values)

    if isinstance(features, list):
        for feature in features:
            df_features[feature] = label_encoder.transform(df_features[feature]) + 1
    else:
        df_features[features] = label_encoder.transform(df_features[features]) + 1
    
    return df_features

def normalize_df(type: str, df: pd.DataFrame) -> pd.DataFrame:
    features = df.columns
    features = list(filter(lambda x: x != "id", features))

    if type == 'StandardScaler':
        scaler = StandardScaler()
    elif type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        raise ValueError("O type informado não é valido!")
    
    df[features] = scaler.fit_transform(df[features])
    return df


if __name__ == "__main__":
    import os
    import sys
    project_root = 'C:/Users/JPedr/OneDrive/Documentos/TCC/Projeto'
    sys.path.append(project_root)

    import pandas as pd
    from app.pipeline.extract import (
        read_data
    )

    PATH ="data\input\Box_Office DataBase(filter).csv"
    df = pd.read_csv(PATH, delimiter=";", encoding="utf-8")
    # print(df)
    df = df.drop('DomesticBox_office', axis=1)
    df = df.drop('InternationalBox_office', axis=1)
    df = df.drop('production_cost', axis=1)
    df = df.drop('Title', axis=1)

    target = df["WorldwideBox_office"]
    df_features = build_target_features(df, "WorldwideBox_office")
    # print(df_features)

    df_features = enconding_features(['Cast_1','Cast_2','Cast_3'],df_features)
    df_features = enconding_features(['Genre_1','Genre_2','Genre_3'],df_features)
    df_features = enconding_features('Director_1',df_features)
    df_features = enconding_features('original_language',df_features)
    df_features = enconding_features('Production_Companies',df_features)
    # print(df_features.columns)

    # df_features = normalize_df('StandardScaler', df_features)
    df_features = normalize_df('MinMaxScaler', df_features)
    print(df_features)
    