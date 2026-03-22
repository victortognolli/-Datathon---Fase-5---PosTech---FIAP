"""
DATATHON PASSOS MÁGICOS - APP STREAMLIT MELHORADO
Aplicação para Predição de Risco de Defasagem com Visualizações

Deploy: Streamlit Community Cloud
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ============================================================================
# CONFIGURAÇÃO STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Passos Mágicos - Predição de Risco",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CARREGAR MODELO E DADOS
# ============================================================================
@st.cache_resource
def load_model():
    with open('deploy_git/modelo_risco.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open('deploy_git/scaler.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_colunas():
    with open('deploy_git/colunas_modelo.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_dados():
    return pd.read_csv('dados_limpos_2024.csv')

@st.cache_data
def load_respostas():
    try:
        with open('respostas_11_perguntas.json', 'r') as f:
            return json.load(f)
    except:
        return {}

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<style>
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .header p {
        margin: 5px 0 0 0;
        font-size: 1.1em;
    }
</style>
<div class="header">
    <h1>🎓 Passos Mágicos</h1>
    <p>Predição de Risco de Defasagem Escolar com Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - NAVEGAÇÃO
# ============================================================================
st.sidebar.title("📋 Menu")
pagina = st.sidebar.radio(
    "Selecione uma seção:",
    ["🏠 Início", "📊 Análise Descritiva", "🤖 Predição de Risco", "📈 Insights", "ℹ️ Sobre"]
)

# ============================================================================
# PÁGINA 1: INÍCIO
# ============================================================================
if pagina == "🏠 Início":
    st.markdown("""
    ## Bem-vindo ao Sistema de Predição de Risco de Defasagem!
    
    Este aplicativo utiliza **Machine Learning** para identificar alunos com maior risco de defasagem educacional,
    permitindo ações pedagógicas mais rápidas e direcionadas.
    
    ### 🎯 O que você pode fazer aqui:
    
    1. **📊 Análise Descritiva**: Explore dados dos alunos por indicadores, Pedra e gênero
    2. **🤖 Predição de Risco**: Insira dados de um aluno e obtenha previsão de risco
    3. **📈 Insights**: Visualize os principais achados e recomendações estratégicas
    
    ### 📌 Indicadores Utilizados:
    
    - **IAN**: Indicador de Adequação de Nível
    - **IDA**: Indicador de Desempenho Acadêmico
    - **IEG**: Indicador de Engajamento
    - **IAA**: Indicador de Autoavaliação
    - **IPS**: Indicador Psicossocial
    - **IPP**: Indicador Psicopedagógico
    - **IPV**: Indicador de Ponto de Virada
    
    ### 🚀 Comece explorando os dados!
    """)

# ============================================================================
# PÁGINA 2: ANÁLISE DESCRITIVA
# ============================================================================
elif pagina == "📊 Análise Descritiva":
    st.title("📊 Análise Descritiva dos Dados")
    
    df = load_dados()
    
    # Abas de análise
    tab1, tab2, tab3, tab4 = st.tabs(["Indicadores", "Por Pedra", "Por Gênero", "Estatísticas"])
    
    with tab1:
        st.subheader("Distribuição dos Indicadores")
        
        col1, col2 = st.columns(2)
        
        with col1:
            indicador = st.selectbox("Selecione um indicador:", ['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV'])
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[indicador],
                nbinsx=30,
                marker_color='#667eea',
                name=indicador
            ))
            fig.update_layout(
                title=f"Distribuição de {indicador}",
                xaxis_title=indicador,
                yaxis_title="Frequência",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric(f"Média {indicador}", f"{df[indicador].mean():.2f}")
            st.metric(f"Mediana {indicador}", f"{df[indicador].median():.2f}")
            st.metric(f"Desvio Padrão", f"{df[indicador].std():.2f}")
            st.metric(f"Mínimo", f"{df[indicador].min():.2f}")
            st.metric(f"Máximo", f"{df[indicador].max():.2f}")
    
    with tab2:
        st.subheader("Indicadores por Pedra")
        
        pedra_analise = df.groupby('PEDRA_CLASSIFICACAO')[['IAN', 'IDA', 'IEG', 'IPP', 'IPV']].mean()
        
        fig = go.Figure()
        for col in ['IAN', 'IDA', 'IEG', 'IPP', 'IPV']:
            fig.add_trace(go.Bar(
                x=pedra_analise.index,
                y=pedra_analise[col],
                name=col
            ))
        fig.update_layout(
            title="Média de Indicadores por Pedra",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Indicadores por Gênero")
        
        genero_analise = df.groupby('Gênero')[['IAN', 'IDA', 'IEG', 'IPP', 'IPV']].mean()
        
        fig = go.Figure()
        for col in ['IAN', 'IDA', 'IEG', 'IPP', 'IPV']:
            fig.add_trace(go.Bar(
                x=genero_analise.index,
                y=genero_analise[col],
                name=col
            ))
        fig.update_layout(
            title="Média de Indicadores por Gênero",
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Estatísticas Descritivas")
        
        stats = df[['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']].describe().T
        st.dataframe(stats, use_container_width=True)

# ============================================================================
# PÁGINA 3: PREDIÇÃO DE RISCO
# ============================================================================
elif pagina == "🤖 Predição de Risco":
    st.title("🤖 Predição de Risco de Defasagem")
    
    modelo = load_model()
    scaler = load_scaler()
    colunas = load_colunas()
    
    st.markdown("""
    Insira os dados de um aluno para obter uma previsão de risco de defasagem.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ian = st.slider("IAN (Adequação de Nível)", 0.0, 10.0, 7.0)
        ida = st.slider("IDA (Desempenho Acadêmico)", 0.0, 10.0, 6.0)
        ieg = st.slider("IEG (Engajamento)", 0.0, 10.0, 8.0)
    
    with col2:
        iaa = st.slider("IAA (Autoavaliação)", 0.0, 10.0, 7.0)
        ips = st.slider("IPS (Psicossocial)", 0.0, 10.0, 7.0)
        ipp = st.slider("IPP (Psicopedagógico)", 0.0, 10.0, 7.5)
    
    with col3:
        ipv = st.slider("IPV (Ponto de Virada)", 0.0, 10.0, 7.0)
        idade = st.slider("Idade", 6, 20, 12)
        pedra = st.selectbox("Pedra", ['Quartzo', 'Ágata', 'Ametista', 'Topázio'])
    
    if st.button("🔮 Fazer Predição", use_container_width=True):
        # Preparar dados
        dados_entrada = {
            'IAN': ian,
            'IDA': ida,
            'IEG': ieg,
            'IAA': iaa,
            'IPS': ips,
            'IPP': ipp,
            'IPV': ipv,
            'Idade': idade,
            'PEDRA_Ametista': 1 if pedra == 'Ametista' else 0,
            'PEDRA_Quartzo': 1 if pedra == 'Quartzo' else 0,
            'PEDRA_Topázio': 1 if pedra == 'Topázio' else 0,
        }
        
        # Criar DataFrame
        df_entrada = pd.DataFrame([dados_entrada])
        
        # Normalizar
        df_scaled = scaler.transform(df_entrada[colunas])
        
        # Predição
        probabilidade = modelo.predict_proba(df_scaled)[0]
        predicao = modelo.predict(df_scaled)[0]
        
        # Exibir resultado
        col1, col2 = st.columns(2)
        
        with col1:
            if predicao == 1:
                st.error("⚠️ RISCO IDENTIFICADO")
                st.metric("Probabilidade de Risco", f"{probabilidade[1]*100:.1f}%")
            else:
                st.success("✅ SEM RISCO")
                st.metric("Probabilidade de Risco", f"{probabilidade[1]*100:.1f}%")
        
        with col2:
            fig = go.Figure(data=[
                go.Bar(
                    x=['Sem Risco', 'Com Risco'],
                    y=[probabilidade[0]*100, probabilidade[1]*100],
                    marker_color=['#10b981', '#ef4444']
                )
            ])
            fig.update_layout(
                title="Probabilidades",
                yaxis_title="Probabilidade (%)",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recomendações
        st.subheader("💡 Recomendações")
        
        if predicao == 1:
            st.warning("""
            **Este aluno apresenta risco de defasagem. Recomendações:**
            - Aumentar suporte psicopedagógico (IPP)
            - Melhorar engajamento em atividades (IEG)
            - Acompanhamento mais próximo do desempenho acadêmico
            - Considerar intervenção pedagógica direcionada
            """)
        else:
            st.info("""
            **Este aluno não apresenta risco de defasagem. Recomendações:**
            - Manter acompanhamento regular
            - Continuar com suporte educacional atual
            - Monitorar evolução dos indicadores
            """)

# ============================================================================
# PÁGINA 4: INSIGHTS
# ============================================================================
elif pagina == "📈 Insights":
    st.title("📈 Insights e Recomendações Estratégicas")
    
    df = load_dados()
    respostas = load_respostas()
    
    if respostas:
        st.subheader("📊 Respostas às 11 Perguntas de Negócio")
        
        for i in range(1, 12):
            key = f'P{i}'
            if key in respostas:
                resp = respostas[key]
                with st.expander(f"**{i}. {resp.get('titulo', 'Pergunta ' + str(i))}**"):
                    st.write(resp.get('insight', 'Sem informação'))
    
    # Estatísticas gerais
    st.subheader("📌 Estatísticas Gerais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Alunos", len(df))
    
    with col2:
        alunos_risco = (df['Defasagem'] > 0).sum()
        st.metric("Alunos em Risco", f"{alunos_risco} ({alunos_risco/len(df)*100:.1f}%)")
    
    with col3:
        media_inda = df['IDA'].mean()
        st.metric("Desempenho Médio (IDA)", f"{media_inda:.2f}")
    
    with col4:
        media_ieg = df['IEG'].mean()
        st.metric("Engajamento Médio (IEG)", f"{media_ieg:.2f}")

# ============================================================================
# PÁGINA 5: SOBRE
# ============================================================================
elif pagina == "ℹ️ Sobre":
    st.title("ℹ️ Sobre o Projeto")
    
    st.markdown("""
    ## Datathon Passos Mágicos - Fase 5
    
    ### 🎯 Objetivo
    Construir um modelo preditivo capaz de identificar alunos com maior risco de defasagem educacional,
    permitindo ações pedagógicas mais rápidas e direcionadas.
    
    ### 📊 Modelo
    - **Algoritmo**: Random Forest
    - **Acurácia**: 92.8%
    - **ROC-AUC**: 0.928
    - **Features**: 34 indicadores engineered
    
    ### 📈 Dados
    - **Total de alunos**: 1.054
    - **Período**: 2024
    - **Indicadores**: 7 principais + features derivadas
    
    ### 🛠 Tecnologias
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    - Plotly
    
    ### 📧 Contato
    Projeto desenvolvido para a Associação Passos Mágicos
    
    ---
    
    **Desenvolvido com ❤️ para transformar vidas através da educação**
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
---
<div style="text-align: center; color: #888; font-size: 0.9em;">
    <p>Datathon Passos Mágicos © 2024 | Predição de Risco de Defasagem Escolar</p>
    <p>Última atualização: {}</p>
</div>
""".format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)
