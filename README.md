# water-potability-mlflow
Projeto Final da disciplina de Tópicos Especiais I (2026.1) - UFC Sobral
### Passo a Passo de Como Executar o Treinamento

* **Link da Base de Dados:** [Kaggle - Water Potability](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
* **Acesso ao Painel MLflow:** http://localhost:5002

---

#### 1. Preparação do Ambiente

1. Abra o seu terminal no VS Code (recomendado utilizar o **CMD** se estiver no Windows).

2. Crie o ambiente virtual (venv):
   ```bash
   python -m venv venv
3. Ative o ambiente virtual:
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
1. Execute o pipeline de MLOps & DataOps. Este arquivo utilizará o Prefect para orquestrar e rodar automaticamente a limpeza, validação (Great Expectations), treino de 5 modelos e a promoção do campeão:
    ```bash
    python scripts/03_orquestrador_prefect.py
  Após a conclusão, o seu navegador abrirá automaticamente o painel de experimentos do MLflow na porta 5002.
  
#### 4. Finalizando os Processos
   1. Para encerrar o servidor temporário do Prefect no terminal, pressione Ctrl + C.
   2. Se o processo do MLflow continuar rodando em segundo plano no Windows (impedindo a limpeza de arquivos), você pode forçar o encerramento executando o comando abaixo em um novo terminal:
      ```bash
      taskkill /F /IM python.exe /T
   3. Para sair do ambiente virtual (venv), digite:
       ```bash
      deactivate
  
