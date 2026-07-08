# Projeto de Automação na Análise de Dados da Crise Global da Água Potável

**Disciplina:** Tópicos Especiais em Computação I (2026.1) - Ciência de Dados  
**Professor:** Iális Cavalcante de Paula Junior  
**Instituição:** Universidade Federal do Ceará - Campus Sobral 

## Equipe - 09
* Gabriela da Silva Melo e Costa (556863)
* Isabela da Silva Melo e Costa (556773)
* Karen Stephan da Penha Sousa (558425)
* Maria Beatriz Vitorino Almeida (554155)



## Sobre o Projeto
De acordo com a Organização Mundial da Saúde (OMS) e a UNICEF, bilhões de pessoas sofrem com a falta de acesso à água segura para consumo. O objetivo deste projeto é construir e avaliar modelos preditivos de classificação supervisionada para determinar se uma amostra de água é potável ou não. Esta solução serve como uma ferramenta de triagem automatizada e de baixo custo para populações vulneráveis.

Utilizamos o dataset **Water Potability** do Kaggle, que contém variáveis químicas como pH, Dureza, Sólidos Dissolvidos, Sulfato, entre outros. O projeto contempla um pipeline completo de DataOps (limpeza e validação) e MLOps (treinamento, rastreamento e registro de modelos).

### Passo a Passo de Como Executar o Treinamento

* **Link da Base de Dados:** [Kaggle - Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
* **Acesso ao Painel MLflow:** http://localhost:5002

---

#### 1. Preparação do Ambiente
Certifique-se de ter o Python 3.10+ instalado.

1. Clone este repositório.
   
2. Abra o seu terminal no VS Code (recomendado utilizar o **CMD** se estiver no Windows).

3. Crie o ambiente virtual (venv):
   ```bash
   python -m venv venv
4. Ative o ambiente virtual:
  * No CMD (Prompt de Comando):
    ```bash
    venv\Scripts\activate.bat
  * No PowerShell:
    ```bash
    .\venv\Scripts\Activate.ps1
5. Instale todas as dependências necessárias:
   ```bash
   python -m pip install -r requirements.txt
  **OBS: Certifique-se de que xgboost e lightgbm estão incluídos no seu arquivo requirements.txt**
  
#### 2. Configuração dos Dados
1. Baixe o arquivo water_potability.csv no link do Kaggle mencionado acima.
2. Crie uma pasta chamada *dados* na raiz do projeto (se já não existir) e cole o arquivo baixado dentro dela. O caminho deve ficar exatamente assim: *dados/water_potability.csv.*
   
#### 3. Execução e Orquestração
**Passo 1: Limpeza e Validação de Dados (DataOps)**
Execute o script abaixo para tratar dados ausentes, remover duplicatas e validar as regras de negócio utilizando a biblioteca Great Expectations.
`python scripts/01_limpar_validar_dados.py`

**Passo 2: Treinamento e Benchmarking (MLOps)**
Execute o script abaixo para separar os dados, aplicar imputação (KNNImputer) e padronização (StandardScaler), e treinar 6 classificadores diferentes (Regressão Logística, Decision Tree, Random Forest, XGBoost, SVM e LightGBM).
`python scripts/02_treinar_modelos_mlflow.py`

**Passo 3: Visualização dos Resultados no MLflow**
Inicie o servidor do MLflow para comparar as métricas dos modelos treinados:
`mlflow ui --backend-store-uri sqlite:///mlflow.db`
Acesse `http://127.0.0.1:5000` no seu navegador.
  
#### 4. Finalizando os Processos
   1. Para encerrar o servidor temporário do Prefect no terminal, pressione Ctrl + C.
   2. Se o processo do MLflow continuar rodando em segundo plano no Windows (impedindo a limpeza de arquivos), você pode forçar o encerramento executando o comando abaixo em um novo terminal:
      ```bash
      taskkill /F /IM python.exe /T
   3. Para sair do ambiente virtual (venv), digite:
       ```bash
      deactivate

## Vídeo de Demonstração
Assista à demonstração completa do funcionamento do projeto e análise dos modelos no link:
**(https://www.youtube.com/watch?v=DjqMyH_w66Q)**
