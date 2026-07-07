import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
os.chdir(RAIZ_PROJETO)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. ATUALIZAÇÃO DO NOME DO MODELO REGISTRADO
NOME_MODELO_REGISTRADO = "Water_Potability_Model"
MINIMA_MELHORIA = 0.0001


def buscar_f1_do_run(client, run_id):
    run = client.get_run(run_id)
    return run.data.metrics.get("f1_score", 0)


def buscar_alias(client, alias):
    try:
        return client.get_model_version_by_alias(
            name=NOME_MODELO_REGISTRADO,
            alias=alias
        )
    except Exception:
        return None


def promover_modelo():
    client = MlflowClient()

    challenger = buscar_alias(client, "challenger")

    if challenger is None:
        print("Nenhum modelo challenger de potabilidade da água foi encontrado.")
        print("Execute primeiro o script de treinamento (02_treinar_modelos_mlflow.py).")
        return

    champion = buscar_alias(client, "champion")

    f1_challenger = buscar_f1_do_run(client, challenger.run_id)

    print(f"Challenger encontrado: versão {challenger.version}")
    print(f"F1-score do challenger: {f1_challenger:.4f}")

    if champion is None:
        print("Nenhum champion existente para potabilidade de água. Promovendo o challenger atual.")

        client.set_registered_model_alias(
            name=NOME_MODELO_REGISTRADO,
            alias="champion",
            version=challenger.version
        )

        client.set_model_version_tag(
            name=NOME_MODELO_REGISTRADO,
            version=challenger.version,
            key="status",
            value="champion"
        )

        print(f"Modelo versão {challenger.version} promovido para champion com sucesso!")
        return

    f1_champion = buscar_f1_do_run(client, champion.run_id)

    print(f"Champion atual em produção: versão {champion.version}")
    print(f"F1-score do champion atual: {f1_champion:.4f}")

    if f1_challenger > f1_champion + MINIMA_MELHORIA:
        print("O modelo Challenger obteve melhor performance. Promovendo para champion...")

        client.set_registered_model_alias(
            name=NOME_MODELO_REGISTRADO,
            alias="champion",
            version=challenger.version
        )

        client.set_model_version_tag(
            name=NOME_MODELO_REGISTRADO,
            version=challenger.version,
            key="status",
            value="champion"
        )

        client.set_model_version_tag(
            name=NOME_MODELO_REGISTRADO,
            version=champion.version,
            key="status",
            value="substituido"
        )

        print(f"Novo champion estabelecido: versão {challenger.version}")

    else:
        print("O Challenger não superou o champion atual significativamente.")
        print("O modelo em produção permanece inalterado.")

        client.set_model_version_tag(
            name=NOME_MODELO_REGISTRADO,
            version=challenger.version,
            key="status",
            value="rejeitado"
        )


if __name__ == "__main__":
    promover_modelo()