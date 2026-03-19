# 🚀 GUIA DE DEPLOY - STREAMLIT COMMUNITY CLOUD

## Passo 1: Preparar Repositório GitHub

### 1.1 Criar Repositório
```bash
git init
git add .
git commit -m "Initial commit - Datathon Passos Mágicos"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/datathon-passos-magicos.git
git push -u origin main
```

### 1.2 Estrutura de Arquivos Necessária
```
datathon-passos-magicos/
├── app_streamlit_melhorado.py      # App principal
├── requirements.txt                 # Dependências
├── modelo_risco.pkl                # Modelo treinado
├── scaler.pkl                      # Scaler
├── colunas_modelo.pkl              # Colunas do modelo
├── dados_limpos_2024.csv           # Dados
├── respostas_11_perguntas.json     # Respostas
└── README.md                       # Documentação
```

## Passo 2: Deploy no Streamlit Community Cloud

### 2.1 Criar Conta
1. Acesse: https://streamlit.io/cloud
2. Clique em "Sign up"
3. Conecte com sua conta GitHub

### 2.2 Fazer Deploy
1. Clique em "New app"
2. Selecione seu repositório
3. Selecione a branch "main"
4. Caminho do arquivo: `app_streamlit_melhorado.py`
5. Clique em "Deploy"

### 2.3 Configurações Importantes
- **App URL**: Será gerada automaticamente
- **Secrets**: Não necessário para este projeto
- **Python Version**: 3.9+

## Passo 3: Verificar Deploy

### 3.1 Acessar App
- URL será: `https://share.streamlit.io/SEU_USUARIO/datathon-passos-magicos/main/app_streamlit_melhorado.py`

### 3.2 Testar Funcionalidades
- [ ] Página Início carrega
- [ ] Análise Descritiva funciona
- [ ] Predição de Risco funciona
- [ ] Insights exibem corretamente
- [ ] Gráficos Plotly renderizam

## Passo 4: Troubleshooting

### Erro: "ModuleNotFoundError"
**Solução**: Adicione o pacote em `requirements.txt`

### Erro: "FileNotFoundError"
**Solução**: Certifique-se que os arquivos .pkl e .csv estão no repositório

### Erro: "Memory exceeded"
**Solução**: Reduza tamanho dos dados ou otimize o modelo

## Passo 5: Manutenção

### Atualizar App
```bash
git add .
git commit -m "Update app"
git push origin main
```
O Streamlit fará redeploy automaticamente.

### Monitorar Performance
- Acesse: https://share.streamlit.io/your-app
- Verifique logs em "Settings" → "Logs"

## Alternativa: Deploy Local

Se preferir testar localmente antes:

```bash
# Instalar dependências
pip install -r requirements.txt

# Rodar app
streamlit run app_streamlit_melhorado.py

# Acessar em http://localhost:8501
```

## URLs Importantes

- **Streamlit Cloud**: https://streamlit.io/cloud
- **Documentação**: https://docs.streamlit.io
- **GitHub**: https://github.com

---

**Status**: ✅ Pronto para Deploy
**Tempo estimado**: 5-10 minutos
