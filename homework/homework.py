# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from glob import glob
from sklearn.neural_network import MLPClassifier
import pandas as pd
import gzip
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    balanced_accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix
)
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#carga el dataset
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path, index_col=False, compression="zip")
    test_df = pd.read_csv(test_path, index_col=False, compression="zip")
    
    for dataframe in [train_df, test_df]:
        dataframe.rename(columns={'default payment next month': "default"}, inplace=True)
        dataframe.drop(columns=["ID"], inplace=True)
        
    return train_df, test_df

#procesamiento del dataset
def preprocess_data(train_df, test_df):
    train_clean = train_df.loc[
        (train_df["MARRIAGE"] != 0) & 
        (train_df["EDUCATION"] != 0)
    ].copy()
    
    test_clean = test_df.loc[
        (test_df["MARRIAGE"] != 0) & 
        (test_df["EDUCATION"] != 0)
    ].copy()
    
    for dataframe in [train_clean, test_clean]:
        dataframe["EDUCATION"] = dataframe["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
        
    return train_clean.dropna(), test_clean.dropna()

#creación del pipeline
def create_model_pipeline(categorical_features, numerical_features):
    """Create preprocessing and model pipeline"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(), categorical_features),
            ('scaling', StandardScaler(), numerical_features),
        ]
    )
    
    return Pipeline([
        ("data_preprocessor", preprocessor),
        ('feature_selector', SelectKBest(score_func=f_classif)),
        ('dim_reduction', PCA()),
        ('mlp_classifier', MLPClassifier(max_iter=15000, random_state=21))
    ])

#calcula las metricas
def calculate_metrics(y_true, y_pred, dataset_type):
    return {
        "type": "metrics",
        "dataset": dataset_type,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


#genera la matriz de confusión
def calculate_confusion_matrix(y_true, y_pred, dataset_type):
    """Calculate and format confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_type,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }


def main():
    
    train_df, test_df = load_data(
        "./files/input/train_data.csv.zip",
        "./files/input/test_data.csv.zip"
    )
    
    train_clean, test_clean = preprocess_data(train_df, test_df)
    
    
    x_train = train_clean.drop(columns=["default"])
    y_train = train_clean["default"]
    x_test = test_clean.drop(columns=["default"])
    y_test = test_clean["default"]
    
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical_cols = [col for col in x_train.columns if col not in categorical_cols]
    
    
    pipeline = create_model_pipeline(categorical_cols, numerical_cols)
    
    
    param_grid = {
        "dim_reduction__n_components": [None],
        "feature_selector__k": [20],
        "mlp_classifier__hidden_layer_sizes": [(50, 30, 40, 60)],
        "mlp_classifier__alpha": [0.26],
        'mlp_classifier__learning_rate_init': [0.001],
    }
    
    
    model = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        refit=True
    )
    
    model.fit(x_train, y_train)
    
    
    if os.path.exists("files/models/"):
        for file in glob(f"files/models/*"):
            os.remove(file)
        os.rmdir("files/models/")
    os.makedirs("files/models/")
    
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(model, f)
    
    
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    
    train_metrics = calculate_metrics(y_train, y_train_pred, "train")
    test_metrics = calculate_metrics(y_test, y_test_pred, "test")
    train_conf = calculate_confusion_matrix(y_train, y_train_pred, "train")
    test_conf = calculate_confusion_matrix(y_test, y_test_pred, "test")
    
    
    os.makedirs("files/output/", exist_ok=True)
    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        for metric in [train_metrics, test_metrics, train_conf, test_conf]:
            file.write(json.dumps(metric) + "\n")

if __name__ == "__main__":
    main()