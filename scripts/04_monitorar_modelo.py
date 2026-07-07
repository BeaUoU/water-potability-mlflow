import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Evita erro de acentuação/Unicode no terminal do Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
os.chdir(RAIZ_PROJETO)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. ATUALIZAÇÃO DO NOME DO EXPERIMENTO
EXPERIMENTO_MONITORAMENTO = "Water_Potability_Monitoramento"

CAMINHO_PREDICOES = Path("dados/predicoes_dashboard.csv")
CAMINHO_ALERTA = Path("dados/alerta_modelo.txt")

LIMITE_F1 = 0.55
AUTO_RETREINAR = True


def imprimir_seguro(texto):
    if not texto:
        return

    try:
        print(texto)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        texto_corrigido = str(texto).encode(
            encoding,
            errors="replace"
        ).decode(
            encoding,
            errors="replace"
        )
        print(texto_corrigido)


def executar_script(caminho_script):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    resultado = subprocess.run(
        [sys.executable, caminho_script],
        cwd=RAIZ_PROJETO,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env
    )

    imprimir_seguro(resultado.stdout)

    if resultado.stderr:
        imprimir_seguro(resultado.stderr)

    if resultado.returncode != 0:
        raise RuntimeError(f"Erro ao executar {caminho_script}")


def carregar_predicoes():
    if not CAMINHO_PREDICOES.exists():
        print("Arquivo de previsões do dashboard ainda não existe.")
        return pd.DataFrame()

    return pd.read_csv(CAMINHO_PREDICOES)


def calcular_metricas(df_avaliado):
    y_real = df_avaliado["resultado_real"].astype(int)
    y_pred = df_avaliado["previsao"].astype(int)

    metricas = {
        "accuracy": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, zero_division=0),
        "recall": recall_score(y_real, y_pred, zero_division=0),
        "f1_score": f1_score(y_real, y_pred, zero_division=0),
    }

    return metricas


def disparar_retreinamento():
    print("\nIniciando retreinamento automático do pipeline de água...")

    executar_script("scripts/02_treinar_modelos_mlflow.py")
    executar_script("scripts/05_promover_modelo.py")

    print("Retreinamento e promoção do novo Champion concluídos.")


def monitorar_modelo():
    df = carregar_predicoes()

    mlflow.set_experiment(EXPERIMENTO_MONITORAMENTO)

    with mlflow.start_run(run_name="monitoramento_modelo"):
        if df.empty:
            mlflow.log_param("status_modelo", "SEM_DADOS")
            print("Sem dados para monitoramento.")
            return

        if "resultado_real" not in df.columns:
            mlflow.log_param("status_modelo", "SEM_RESULTADO_REAL")
            print("A coluna resultado_real não existe nas predições do dashboard.")
            return

        df_avaliado = df.dropna(subset=["resultado_real"]).copy()

        total_predicoes = len(df)
        total_avaliadas = len(df_avaliado)

        mlflow.log_metric("total_predicoes_dashboard", total_predicoes)
        mlflow.log_metric("total_amostras_avaliadas", total_avaliadas)

        if total_avaliadas == 0:
            mlflow.log_param("status_modelo", "SEM_RESULTADOS_REAIS")
            print("Ainda não há nenhuma amostra real de água avaliada em laboratório.")
            return

        metricas = calcular_metricas(df_avaliado)

        # 2. ADAPTAÇÃO DAS MÉTRICAS DE PRODUÇÃO PARA POTABILIDADE DA ÁGUA
        probabilidade_media = df["probabilidade_potabilidade"].mean()
        taxa_potavel = df["previsao"].mean()

        mlflow.log_metric("production_accuracy", metricas["accuracy"])
        mlflow.log_metric("production_precision", metricas["precision"])
        mlflow.log_metric("production_recall", metricas["recall"])
        mlflow.log_metric("production_f1_score", metricas["f1_score"])
        mlflow.log_metric("probabilidade_media_potabilidade", probabilidade_media)
        mlflow.log_metric("taxa_agua_classificada_potavel", taxa_potavel)

        print("Métricas de produção observadas:")
        print(f"Amostras avaliadas: {total_avaliadas}")
        print(f"Acurácia: {metricas['accuracy']:.4f}")
        print(f"Precisão: {metricas['precision']:.4f}")
        print(f"Recall: {metricas['recall']:.4f}")
        print(f"F1-score: {metricas['f1_score']:.4f}")

        if metricas["f1_score"] < LIMITE_F1:
            status = "RUIM"

            mensagem = (
                f"ALERTA: modelo de potabilidade abaixo do limite de segurança.\n"
                f"F1-score atual: {metricas['f1_score']:.4f}\n"
                f"Limite mínimo aceitável: {LIMITE_F1:.4f}\n"
                f"Amostras avaliadas: {total_avaliadas}\n"
            )

            CAMINHO_ALERTA.write_text(mensagem, encoding="utf-8")

            mlflow.log_param("status_modelo", status)
            mlflow.log_artifact(str(CAMINHO_ALERTA))

            print(mensagem)

            if AUTO_RETREINAR:
                mlflow.log_param("retreinamento_acionado", "SIM")
                disparar_retreinamento()
            else:
                mlflow.log_param("retreinamento_acionado", "NAO")

        else:
            status = "BOM"

            mlflow.log_param("status_modelo", status)
            mlflow.log_param("retreinamento_acionado", "NAO")

            print("Modelo operando estavelmente dentro das métricas esperadas.")


if __name__ == "__main__":
    monitorar_modelo()