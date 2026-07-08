import os
from pathlib import Path

import pandas as pd
import great_expectations as gx


RAIZ_PROJETO = Path(__file__).resolve().parents[1]
os.chdir(RAIZ_PROJETO)

CAMINHO_CSV = Path("dados/water_potability.csv")
CAMINHO_SAIDA = Path("dados/water_potability_limpo.csv")

COLUNAS_OBRIGATORIAS = [
    "ph",
    "Hardness",
    "Solids",
    "Chloramines",
    "Sulfate",
    "Conductivity",
    "Organic_carbon",
    "Trihalomethanes",
    "Turbidity",
    "Potability"
]


def carregar_dados():
    if not CAMINHO_CSV.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {CAMINHO_CSV}. "
            "Certifique-se de salvar o arquivo 'water_potability.csv' dentro da pasta dados."
        )

    df = pd.read_csv(CAMINHO_CSV)
    return df


def validar_colunas_originais(df):
    colunas_faltando = [coluna for coluna in COLUNAS_OBRIGATORIAS if coluna not in df.columns]

    if colunas_faltando:
        raise ValueError(f"Colunas obrigatórias ausentes: {colunas_faltando}")


def limpar_dados(df):
    df = df.copy()

    # Remove espaços extras nos nomes das colunas
    df.columns = df.columns.str.strip()

    # Remove linhas completamente duplicadas
    df = df.drop_duplicates()

    # Garante que todas as colunas de medição sejam numéricas float
    colunas_medicao = [col for col in COLUNAS_OBRIGATORIAS if col != "Potability"]
    for col in colunas_medicao:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Garante que a potabilidade seja convertida antes do descarte total de nulos
    df["Potability"] = pd.to_numeric(df["Potability"], errors="coerce")

     # Como não dropamos os nulos da tabela inteira, dropamos APENAS se o alvo (Potability) for nulo
    df = df.dropna(subset=["Potability"])

    # Reseta o índice para que a validação do Great Expectations não se confunda com índices desalinhados
    df = df.reset_index(drop=True)

    # Converte explicitamente a potabilidade para inteiro
    df["Potability"] = df["Potability"].astype(int)

    return df


def validar_com_great_expectations(df):
    context = gx.get_context(mode="ephemeral")

    data_source = context.data_sources.add_pandas("fonte_pandas_agua")
    data_asset = data_source.add_dataframe_asset(name="agua_limpa")

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "batch_agua_limpa"
    )

    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    expectativas = [
        # Esperamos exatamente 10 colunas na saída
        gx.expectations.ExpectTableColumnCountToEqual(value=10),

        # Validando se a coluna alvo existe e está correta
        gx.expectations.ExpectColumnToExist(column="Potability"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="Potability"),
        gx.expectations.ExpectColumnValuesToBeInSet(column="Potability", value_set=[0, 1]),

        # Regra de Negócio: O pH da água deve estar idealmente em uma escala de 0 a 14
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="ph",
            min_value=0.0,
            max_value=14.0
        ),

        # Regra de Negócio: Condutividade e Outros elementos não podem ser negativos
        gx.expectations.ExpectColumnValuesToBeBetween(column="Conductivity", min_value=0.0),
        gx.expectations.ExpectColumnValuesToBeBetween(column="Sulfate", min_value=0.0),
        gx.expectations.ExpectColumnValuesToBeBetween(column="Solids", min_value=0.0),
        
        
    ]

    erros = []

    for expectativa in expectativas:
        resultado = batch.validate(expectativa)

        if not resultado.success:
            erros.append(str(expectativa))

    if erros:
        print("\nA base falhou na validação de qualidade da água (Great Expectations):")
        for erro in erros:
            print("-", erro)

        raise ValueError("Os dados de água não passaram nas validações de qualidade de DataOps.")

    print("A base de água passou com sucesso em todas as validações do Great Expectations.")



def salvar_dados_limpos(df):
    Path("dados").mkdir(exist_ok=True)
    df.to_csv(CAMINHO_SAIDA, index=False)


def main():
    print("Carregando dados...")
    df = carregar_dados()

    print(f"Base original: {df.shape[0]} linhas e {df.shape[1]} colunas")

    print("Validando colunas originais...")
    validar_colunas_originais(df)

    print("Limpando dados...")
    df_limpo = limpar_dados(df)

    print(f"Base limpa: {df_limpo.shape[0]} linhas e {df_limpo.shape[1]} colunas")

    print("Validando dados com Great Expectations...")
    validar_com_great_expectations(df_limpo)

    print("Salvando base limpa...")
    salvar_dados_limpos(df_limpo)

    print(f"Arquivo criado: {CAMINHO_SAIDA}")
    print("Etapa DataOps concluída com sucesso.")


if __name__ == "__main__":
    main()