# import os
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import streamlit as st
# import mlflow
# import mlflow.sklearn

# # Configuração da página do Streamlit
# st.set_page_config(
#     page_title="DataOps & MLOps - Qualidade da Água",
#     page_icon="💧",
#     layout="wide"
# )

# # --- Configuração da Barra Lateral (Sidebar) ---
# #st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3105/3105807.png", width=100) # Ícone de gota d'água
# st.sidebar.title("💧 Menu de Controle")
# st.sidebar.markdown("""
# Este painel monitora a **Qualidade e Potabilidade da Água** em tempo real utilizando modelos preditivos de Machine Learning integrados via **MLflow** e orquestrados por **Prefect**.
# """)

# st.sidebar.info("📌 Status do Modelo: Conectado ao Model Registry")

# RAIZ_PROJETO = Path(__file__).resolve().parent
# os.chdir(RAIZ_PROJETO)

# # Configura o MLflow para ler o banco local
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# NOME_MODELO_REGISTRADO = "Water_Potability_Model"
# CAMINHO_PREDICOES = Path("dados/predicoes_dashboard.csv")
# CAMINHO_ALERTA = Path("dados/alerta_modelo.txt")

# # Função para carregar o modelo Champion do MLflow
# @st.cache_resource
# def carregar_modelo_champion():
#     try:
#         modelo_uri = f"models://{NOME_MODELO_REGISTRADO}@champion"
#         modelo = mlflow.sklearn.load_model(modelo_uri)
#         return modelo
#     except Exception as e:
#         return None

# # Carrega o modelo
# modelo_champion = carregar_modelo_champion()

# # Título do Dashboard
# st.title("💧 Sistema de Monitoramento e Potabilidade da Água")
# st.subheader("Painel de DataOps & MLOps em Tempo Real")

# # --- Painel de Alerta do Modelo ---
# if CAMINHO_ALERTA.exists():
#     conteudo_alerta = CAMINHO_ALERTA.read_text(encoding="utf-8")
#     st.error(f"⚠️ **Alerta de Degradação detectado pelo loop de monitoramento:**\n\n{conteudo_alerta}")

# # --- SEÇÃO 1: Estatísticas Gerais das Predições Realizadas ---
# st.markdown("---")
# st.header("📊 Métricas de Operação (Dashboard)")

# if CAMINHO_PREDICOES.exists():
#     df_historico = pd.read_csv(CAMINHO_PREDICOES)
#     total_amostras = len(df_historico)
    
#     if total_amostras > 0:
#         # Se houver coluna de previsão, calcula a taxa de potabilidade
#         taxa_potavel = (df_historico["previsao"].sum() / total_amostras) * 100 if "previsao" in df_historico.columns else 0
#         ph_medio = df_historico["ph"].mean() if "ph" in df_historico.columns else 7.0
        
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Total de Amostras Analisadas", f"{total_amostras}")
#         col2.metric("Taxa de Água Potável", f"{taxa_potavel:.1f}%")
#         col3.metric("pH Médio das Amostras", f"{ph_medio:.2f}")
        
#         # Histórico recente
#         st.subheader("📋 Últimas Análises Registradas")
#         st.dataframe(df_historico.tail(5), use_container_width=True)
#     else:
#         st.info("O arquivo de predições está vazio. Realize uma simulação abaixo.")
# else:
#     st.info("Nenhuma predição foi realizada ainda. Use o simulador abaixo para gerar dados.")

# # --- SEÇÃO 2: Simulador em Tempo Real (Formulário de Entrada) ---
# st.markdown("---")
# st.header("🔬 Simulador de Análise Química da Água")

# if modelo_champion is None:
#     st.warning("⚠️ Nenhum modelo 'champion' foi encontrado no MLflow. Execute os scripts de treinamento e promoção primeiro.")
# else:
#     st.success("✅ Modelo Produtivo (Champion) carregado com sucesso do Model Registry!")
    
#     with st.form("formulario_agua"):
#         col_a, col_b, col_c = st.columns(3)
        
#         with col_a:
#             ph = st.slider("pH (Acidez/Alcalinidade)", 0.0, 14.0, 7.0, step=0.1)
#             Hardness = st.number_input("Dureza (Hardness - mg/L)", min_value=0.0, value=200.0)
#             Solids = st.number_input("Sólidos Totais Dissolvidos (ppm)", min_value=0.0, value=20000.0)
            
#         with col_b:
#             Chloramines = st.number_input("Cloraminas (ppm)", min_value=0.0, value=7.0)
#             Sulfate = st.number_input("Sulfato (mg/L)", min_value=0.0, value=330.0)
#             Conductivity = st.number_input("Condutividade (μS/cm)", min_value=0.0, value=420.0)
            
#         with col_c:
#             Organic_carbon = st.number_input("Carbono Orgânico Total (mg/L)", min_value=0.0, value=14.0)
#             Trihalomethanes = st.number_input("Trihalometanos (μg/L)", min_value=0.0, value=65.0)
#             Turbidity = st.slider("Turbidez (NTU)", 0.0, 10.0, 4.0, step=0.1)
            
#         # Campo opcional para simular a validação laboratorial real depois (usado pelo script 04)
#         resultado_real = st.selectbox(
#             "Resultado Real de Laboratório (Opcional - Usado para testar o Monitoramento)", 
#             options=[np.nan, 1, 0], 
#             format_func=lambda x: "Não avaliado ainda" if pd.isna(x) else ("Potável" if x == 1 else "Insegura")
#         )

#         botao_enviar = st.form_submit_button("⚡ Analisar Potabilidade")
        
#     if botao_enviar:
#         # Monta o DataFrame com as colunas EXATAS que o modelo espera
#         dados_entrada = pd.DataFrame([{
#             "ph": ph,
#             "Hardness": Hardness,
#             "Solids": Solids,
#             "Chloramines": Chloramines,
#             "Sulfate": Sulfate,
#             "Conductivity": Conductivity,
#             "Organic_carbon": Organic_carbon,
#             "Trihalomethanes": Trihalomethanes,
#             "Turbidity": Turbidity
#         }])
        
#         # Realiza as predições usando o pipeline do Scikit-Learn vindo do MLflow
#         previsao = int(modelo_champion.predict(dados_entrada)[0])
#         # Pega a probabilidade da classe 1 (Potável)
#         probabilidade = float(modelo_champion.predict_proba(dados_entrada)[0][1])
        
#         # Exibe o resultado na tela
#         st.markdown("### 📋 Resultado da Avaliação:")
#         if previsao == 1:
#             st.balloons()
#             st.success(f"🟢 **ÁGUA POTÁVEL** (Confiança: {probabilidade*100:.1f}%) - Própria para consumo humano de acordo com os parâmetros estatísticos.")
#         else:
#             st.error(f"🔴 **ÁGUA IMPRÓPRIA / INSEGURA** (Probabilidade de Potabilidade: {probabilidade*100:.1f}%) - Risco de contaminação ou desvio dos padrões.")
            
#         # Salva a predição no CSV histórico (essencial para o script 04_monitorar_modelo.py funcionar!)
#         dados_entrada["previsao"] = previsao
#         dados_entrada["probabilidade_potabilidade"] = probabilidade
#         dados_entrada["resultado_real"] = resultado_real
        
#         # Se o arquivo já existe, anexa. Se não, cria um novo.
#         Path("dados").mkdir(exist_ok=True)
#         if CAMINHO_PREDICOES.exists():
#             df_existente = pd.read_csv(CAMINHO_PREDICOES)
#             df_final = pd.concat([df_existente, dados_entrada], ignore_index=True)
#         else:
#             df_final = dados_entrada
            
#         df_final.to_csv(CAMINHO_PREDICOES, index=False)
#         st.info("💾 Registro da amostra salvo na base de predições. Atualize a página para recalcular os gráficos.")