import os
import time
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configuração de caminhos e ambiente
RAIZ_PROJETO = Path(__file__).resolve().parents[1]
os.chdir(RAIZ_PROJETO)

# Banco local do MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Atualização de caminhos e nomes de experimentos
CAMINHO_DADOS = "dados/water_potability_limpo.csv"
CAMINHO_RESULTADOS = "dados/resultados_modelos_agua.csv"

NOME_EXPERIMENTO = "Water_Potability_MLOps"
NOME_MODELO_REGISTRADO = "Water_Potability_Model"


def carregar_dados():
    if not Path(CAMINHO_DADOS).exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {CAMINHO_DADOS}. "
            "Execute primeiro o script de limpeza e validação de dados."
        )
    return pd.read_csv(CAMINHO_DADOS)


def separar_dados(df):
    
    X = df.drop(columns=["Potability"])
    y = df["Potability"]

    # Identificando colunas numéricas
    colunas_numericas = X.select_dtypes(include=["int64", "float64", "int32"]).columns

    # Aplicando o Escalador em todas as colunas numéricas de forma explícita
    pre_processador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), colunas_numericas),
        ]
    )

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Mantém a proporção de água potável/não potável equilibrada
    )

    return X_treino, X_teste, y_treino, y_teste, pre_processador


def calcular_metricas(y_teste, previsoes):
    return {
        "accuracy": accuracy_score(y_teste, previsoes),
        "precision": precision_score(y_teste, previsoes, zero_division=0),
        "recall": recall_score(y_teste, previsoes, zero_division=0),
        "f1_score": f1_score(y_teste, previsoes, zero_division=0),
    }


def treinar_e_registrar_modelo(
    nome_modelo,
    modelo,
    pre_processador,
    X_treino,
    X_teste,
    y_treino,
    y_teste
):
    pipeline = Pipeline(
        steps=[
            ("pre_processamento", pre_processador),
            ("modelo", modelo),
        ]
    )

    with mlflow.start_run(run_name=nome_modelo) as run:
        pipeline.fit(X_treino, y_treino)

        previsoes = pipeline.predict(X_teste)
        metricas = calcular_metricas(y_teste, previsoes)

        mlflow.log_param("modelo", nome_modelo)
        mlflow.log_param("total_linhas_treino", len(X_treino))
        mlflow.log_param("total_linhas_teste", len(X_teste))

        for nome_parametro, valor_parametro in modelo.get_params().items():
            # Filtra parâmetros muito longos para evitar quebras no MLflow local
            if len(str(valor_parametro)) < 250:
                mlflow.log_param(nome_parametro, valor_parametro)

        for nome_metrica, valor in metricas.items():
            mlflow.log_metric(nome_metrica, valor)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="modelo",
            input_example=X_teste.head(3).copy(),
            serialization_format="cloudpickle"
        )

        print(f"\nModelo: {nome_modelo}")
        print(f"Acurácia: {metricas['accuracy']:.4f}")
        print(f"Precisão: {metricas['precision']:.4f}")
        print(f"Recall: {metricas['recall']:.4f}")
        print(f"F1-score: {metricas['f1_score']:.4f}")

        return {
            "modelo": nome_modelo,
            "run_id": run.info.run_id,
            **metricas
        }


def registrar_melhor_como_challenger(melhor_run_id):
    client = MlflowClient()
    modelo_uri = f"runs:/{melhor_run_id}/modelo"

    print("\nRegistrando melhor modelo no Model Registry...")
    resultado = mlflow.register_model(
        model_uri=modelo_uri,
        name=NOME_MODELO_REGISTRADO
    )

    versao = resultado.version

    # Aguarda o modelo estar pronto no registro
    for _ in range(30):
        model_version = client.get_model_version(
            name=NOME_MODELO_REGISTRADO,
            version=versao
        )
        if model_version.status == "READY":
            break
        time.sleep(1)

    client.set_registered_model_alias(
        name=NOME_MODELO_REGISTRADO,
        alias="challenger",
        version=versao
    )

    client.set_model_version_tag(
        name=NOME_MODELO_REGISTRADO,
        version=versao,
        key="status",
        value="challenger"
    )

    print(f"Modelo registrado como challenger. Versão: {versao}")
    return versao


def main():
    print("Carregando dados limpos de potabilidade...")
    df = carregar_dados()

    print(f"Base carregada: {df.shape[0]} linhas e {df.shape[1]} colunas")

    X_treino, X_teste, y_treino, y_teste, pre_processador = separar_dados(df)

    mlflow.set_experiment(NOME_EXPERIMENTO)

    modelos = {
        "Regressao_Logistica": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Arvore_Decisao": DecisionTreeClassifier(
            random_state=42,
            max_depth=6
        ),
        "Random_Forest": RandomForestClassifier(
            random_state=42,
            n_estimators=150,
            max_depth=10
        ),
    }

    resultados = []

    for nome_modelo, modelo in modelos.items():
        resultado = treinar_e_registrar_modelo(
            nome_modelo=nome_modelo,
            modelo=modelo,
            pre_processador=pre_processador,
            X_treino=X_treino,
            X_teste=X_teste,
            y_treino=y_treino,
            y_teste=y_teste
        )
        resultados.append(resultado)

    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(
        by="f1_score",
        ascending=False
    )

    print("\nResumo dos modelos:")
    print(df_resultados[["modelo", "accuracy", "precision", "recall", "f1_score"]])

    melhor_modelo = df_resultados.iloc[0]

    print("\nMelhor modelo pelo F1-score:")
    print(melhor_modelo["modelo"])
    print(f"F1-score: {melhor_modelo['f1_score']:.4f}")

   
    Path("dados").mkdir(exist_ok=True)
    df_resultados.to_csv(CAMINHO_RESULTADOS, index=False)

    print(f"\nArquivo criado: {CAMINHO_RESULTADOS}")

    registrar_melhor_como_challenger(melhor_modelo["run_id"])

    print("Treinamento concluído e salvo no MLflow local com sucesso!")


if __name__ == "__main__":
    main()