import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

def main():
    # 1. Recebendo os argumentos de entrada (Dados e Hiperparâmetros do AG)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Caminho para o arquivo CSV de dados")
    parser.add_argument("--n_estimators", type=int, default=100, help="Número de árvores")
    parser.add_argument("--max_depth", type=int, default=10, help="Profundidade máxima")
    parser.add_argument("--min_samples_split", type=float, default=0.05, help="Mínimo de amostras para divisão")
    args = parser.parse_args()

    # 2. Iniciando o rastreamento do MLflow
    with mlflow.start_run() as run:
        # Autologging ativado: O MLflow vai salvar hiperparâmetros e a árvore automaticamente
        mlflow.sklearn.autolog()

        print(f"Carregando dados de: {args.dataset}")
        df = pd.read_csv(args.dataset)

        # Limpeza idêntica ao seu notebook
        df = df.drop(columns=['Unnamed: 32', 'id'], errors='ignore')
        if df['diagnosis'].dtype == 'object':
            df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

        # Separando X (features) e y (target)
        X = df.drop('diagnosis', axis=1)
        y = df['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Configurando o modelo com os melhores hiperparâmetros do seu AG
        print("Configurando o modelo com os hiperparâmetros otimizados...")
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            random_state=42
        )

        print("Iniciando treinamento da Junta Médica...")
        clf.fit(X_train, y_train)

        # 4. Avaliando o modelo
        print("Avaliando o modelo...")
        y_pred = clf.predict(X_test)

        # Métricas clínicas cruciais para o diagnóstico
        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        
        # 5. Gerando a Matriz de Confusão Visual
        print("Gerando Matriz de Confusão...")
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, 
            display_labels=['Benigno (0)', 'Maligno (1)'], 
            cmap="Blues"
        )
        plt.title("Matriz de Confusão - Diagnóstico de Câncer")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # 6. Registrando o modelo no workspace
        print("Registrando o modelo oficial...")
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="modelo_diagnostico_cancer",
            registered_model_name="Modelo_Mama_AG_Oficial"
        )

        print("✅ Experimento finalizado e modelo registrado com sucesso!")

if __name__ == "__main__":
    main()