import subprocess
import sys
import time
from pathlib import Path


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
INTERVALO_SEGUNDOS = 300


def executar_monitoramento():
    print("\nExecutando monitoramento automático...")

    resultado = subprocess.run(
        [sys.executable, "scripts/04_monitorar_modelo.py"],
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
        print("Erro durante o monitoramento.")
    else:
        print("Monitoramento concluído.")


def main():
    print("Serviço de monitoramento automático iniciado.")

    while True:
        executar_monitoramento()
        print(f"Aguardando {INTERVALO_SEGUNDOS} segundos para nova verificação...")
        time.sleep(INTERVALO_SEGUNDOS)


if __name__ == "__main__":
    main()