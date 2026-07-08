import os
import time
from pathlib import Path
import subprocess
import sys
import socket
import webbrowser
import shutil

from prefect import flow, task

# Configuração de caminhos do projeto
RAIZ_PROJETO = Path(__file__).resolve().parents[1]

PORTA_MLFLOW = 5002
# PORTA_STREAMLIT = 8501

ARQUIVOS_PARA_LIMPAR = [
    "mlflow.db",
    "dados/water_potability_limpo.csv",
    "dados/resultados_modelos_agua.csv",
    "dados/predicoes_dashboard.csv",
    "dados/alerta_modelo.txt",
]

PASTAS_PARA_LIMPAR = [
    "mlruns",
]


def porta_em_uso(porta):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", porta)) == 0


def executar_script(caminho_script):
    caminho_completo = RAIZ_PROJETO / caminho_script

    resultado = subprocess.run(
        [sys.executable, str(caminho_completo)],
        cwd=RAIZ_PROJETO,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace"
    )

    print(resultado.stdout)

    if resultado.stderr:
        print(resultado.stderr)

    if resultado.returncode != 0:
        raise RuntimeError(f"Erro ao executar script: {caminho_script}")


@task(name="Limpar execução anterior")
def limpar_execucao_anterior():
    print("Limpando artefatos e bases de dados de execuções anteriores...")
    
    # Remove as pastas de runs (mlruns)
    for pasta_nome in PASTAS_PARA_LIMPAR:
        pasta = RAIZ_PROJETO / pasta_nome
        if pasta.exists() and pasta.is_dir():
            try:
                shutil.rmtree(pasta)
            except Exception as e:
                print(f"Aviso: Não foi possível deletar a pasta {pasta_nome}: {e}")

    # Remove os arquivos CSVs e o arquivo de alerta
    for arquivo_nome in ARQUIVOS_PARA_LIMPAR:
        caminho = RAIZ_PROJETO / arquivo_nome
        if caminho.exists() and caminho.is_file():
            try:
                # Tratamento especial para o banco SQLite travado pelo Windows
                if arquivo_nome == "mlflow.db":
                    # Ao invés de deletar o arquivo físico, tentamos limpá-lo abrindo-o em modo de escrita vazio
                    with open(caminho, "w"): 
                        pass
                    print("Banco mlflow.db reiniciado/esvaziado com sucesso.")
                else:
                    caminho.unlink()
            except PermissionError:
                print(f"Aviso: O arquivo '{arquivo_nome}' está em uso por outro processo e não pôde ser deletado agora. O pipeline continuará assim mesmo.")
            except Exception as e:
                print(f"Erro ao tentar remover {arquivo_nome}: {e}")


@task(name="Limpeza e validação dos dados", retries=1)
def etapa_limpeza_validacao():
    executar_script("scripts/01_limpar_validar_dados.py")


@task(name="Treinamento e registro no MLflow", retries=1)
def etapa_treinamento_mlflow():
    executar_script("scripts/02_treinar_modelos_mlflow.py")


@task(name="Promoção do modelo Champion", retries=0)
def etapa_promocao_modelo():
    executar_script("scripts/05_promover_modelo.py")


@task(name="Iniciar MLflow")
def iniciar_mlflow():
    if porta_em_uso(PORTA_MLFLOW):
        print(f"O servidor do MLflow já está ativo na porta {PORTA_MLFLOW}.")
        return

    print("Iniciando servidor local do MLflow...")

    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            "sqlite:///mlflow.db",
            "--default-artifact-root",
            "./mlruns",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORTA_MLFLOW),
        ],
        cwd=RAIZ_PROJETO
    )

    time.sleep(5)
    webbrowser.open(f"http://127.0.0.1:{PORTA_MLFLOW}")

# nao sera necessario mais iniciar o streamlit, pois o dashboard sera iniciado pelo mlflow
# @task(name="Iniciar Dashboard Streamlit")
# def iniciar_streamlit():
#     if porta_em_uso(PORTA_STREAMLIT):
#         print(f"O Dashboard Streamlit já está ativo na porta {PORTA_STREAMLIT}.")
#         return

#     print("Iniciando interface do dashboard Streamlit...")

#     subprocess.Popen(
#         [
#             sys.executable,
#             "-m",
#             "streamlit",
#             "run",
#             "dashboard.py",
#             "--server.address",
#             "127.0.0.1",
#             "--server.port",
#             str(PORTA_STREAMLIT),
#         ],
#         cwd=RAIZ_PROJETO
#     )

#     time.sleep(5)
#     webbrowser.open(f"http://127.0.0.1:{PORTA_STREAMLIT}")


@flow(name="Pipeline de Água - DataOps e MLOps")
def pipeline_agua():
    limpar_execucao_anterior()

    etapa_limpeza_validacao()
    etapa_treinamento_mlflow()
    etapa_promocao_modelo()

    iniciar_mlflow()
    # iniciar_streamlit()

    print("\n Pipeline DataOps/MLOps de potabilidade da água concluído com sucesso!")
    print(f"Acesse o MLflow local em: http://127.0.0.1:{PORTA_MLFLOW}")
    # print(f"Acesse o Dashboard em: http://127.0.0.1:{PORTA_STREAMLIT}")


@flow(name="Pipeline Monitoramento de qualidade da agua")
def pipeline_monitoramento():
    executar_script("scripts/04_monitorar_modelo.py")


if __name__ == "__main__":
    pipeline_agua()