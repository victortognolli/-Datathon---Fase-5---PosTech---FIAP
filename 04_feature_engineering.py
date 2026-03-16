import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("FASE 4: FEATURE ENGINEERING - DATATHON PASSOS MÁGICOS")
print("=" * 100)

# ============================================================================
# 1. CARREGAR DADOS LIMPOS
# ============================================================================
print("\n[1/6] Carregando dados limpos...")

df = pd.read_csv('/home/ubuntu/dados_limpos_2024.csv')

print(f"✓ Base carregada: {df.shape[0]} alunos, {df.shape[1]} colunas")
print(f"✓ Variável alvo (RISCO_DEFASAGEM): {df['RISCO_DEFASAGEM'].sum()} em risco")

# ============================================================================
# 2. CRIAR FEATURES DERIVADAS
# ============================================================================
print("\n[2/6] Criando features derivadas...")

# Copiar dataframe para não alterar original
df_features = df.copy()

# 2.1 - Features de Indicadores Agregados
print("\n  2.1 - Features de Agregação:")

# Média dos indicadores
df_features['MEDIA_INDICADORES'] = df_features[['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']].mean(axis=1)
print(f"    ✓ MEDIA_INDICADORES")

# Desvio padrão dos indicadores (variabilidade)
df_features['STD_INDICADORES'] = df_features[['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']].std(axis=1)
print(f"    ✓ STD_INDICADORES")

# Mínimo dos indicadores (pior dimensão)
df_features['MIN_INDICADORES'] = df_features[['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']].min(axis=1)
print(f"    ✓ MIN_INDICADORES")

# Máximo dos indicadores (melhor dimensão)
df_features['MAX_INDICADORES'] = df_features[['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']].max(axis=1)
print(f"    ✓ MAX_INDICADORES")

# 2.2 - Features de Combinações Importantes
print("\n  2.2 - Features de Combinações:")

# Combinação IPP + IDA (psicopedagogia + desempenho)
df_features['IPP_IDA_MEDIA'] = (df_features['IPP'] + df_features['IDA']) / 2
print(f"    ✓ IPP_IDA_MEDIA")

# Combinação IPP + IEG (psicopedagogia + engajamento)
df_features['IPP_IEG_MEDIA'] = (df_features['IPP'] + df_features['IEG']) / 2
print(f"    ✓ IPP_IEG_MEDIA")

# Combinação IDA + IEG (desempenho + engajamento)
df_features['IDA_IEG_MEDIA'] = (df_features['IDA'] + df_features['IEG']) / 2
print(f"    ✓ IDA_IEG_MEDIA")

# Combinação dos 3 principais preditores
df_features['TRIO_PRINCIPAL'] = (df_features['IPP'] + df_features['IDA'] + df_features['IEG']) / 3
print(f"    ✓ TRIO_PRINCIPAL")

# 2.3 - Features de Relação com Risco
print("\n  2.3 - Features de Relação com Risco:")

# Distância do IAN em relação à mediana (10.0)
df_features['DISTANCIA_IAN_MEDIANA'] = abs(df_features['IAN'] - 10.0)
print(f"    ✓ DISTANCIA_IAN_MEDIANA")

# Indicador de baixo desempenho (IDA < 5)
df_features['BAIXO_DESEMPENHO'] = (df_features['IDA'] < 5).astype(int)
print(f"    ✓ BAIXO_DESEMPENHO")

# Indicador de baixo engajamento (IEG < 6)
df_features['BAIXO_ENGAJAMENTO'] = (df_features['IEG'] < 6).astype(int)
print(f"    ✓ BAIXO_ENGAJAMENTO")

# Indicador de baixo suporte psicopedagógico (IPP < 6)
df_features['BAIXO_IPP'] = (df_features['IPP'] < 6).astype(int)
print(f"    ✓ BAIXO_IPP")

# Contagem de indicadores baixos
df_features['CONTAGEM_BAIXOS'] = (
    df_features['BAIXO_DESEMPENHO'] + 
    df_features['BAIXO_ENGAJAMENTO'] + 
    df_features['BAIXO_IPP']
)
print(f"    ✓ CONTAGEM_BAIXOS")

# 2.4 - Features de Evolução Temporal
print("\n  2.4 - Features de Evolução Temporal:")

# Converter INDE histórico para numérico
df_features['INDE_22_NUM'] = pd.to_numeric(df_features['INDE 22'], errors='coerce')
df_features['INDE_23_NUM'] = pd.to_numeric(df_features['INDE 23'], errors='coerce')
df_features['INDE_2024_NUM'] = df_features['INDE_2024_CLEAN']

# Variação 2022-2023
df_features['VARIACAO_22_23'] = df_features['INDE_23_NUM'] - df_features['INDE_22_NUM']
print(f"    ✓ VARIACAO_22_23")

# Variação 2023-2024
df_features['VARIACAO_23_24'] = df_features['INDE_2024_NUM'] - df_features['INDE_23_NUM']
print(f"    ✓ VARIACAO_23_24")

# Tendência geral (2022-2024)
df_features['TENDENCIA_GERAL'] = df_features['INDE_2024_NUM'] - df_features['INDE_22_NUM']
print(f"    ✓ TENDENCIA_GERAL")

# Indicador de deterioração (queda em 2024)
df_features['DETERIORACAO_2024'] = (df_features['VARIACAO_23_24'] < -0.5).astype(int)
print(f"    ✓ DETERIORACAO_2024")

# 2.5 - Features de Contexto
print("\n  2.5 - Features de Contexto:")

# Idade normalizada
df_features['IDADE_NORMALIZADA'] = (df_features['Idade'] - df_features['Idade'].min()) / (df_features['Idade'].max() - df_features['Idade'].min())
print(f"    ✓ IDADE_NORMALIZADA")

# Anos na PM normalizado
df_features['ANOS_NA_PM_NORMALIZADO'] = (df_features['ANOS_NA_PM'] - df_features['ANOS_NA_PM'].min()) / (df_features['ANOS_NA_PM'].max() - df_features['ANOS_NA_PM'].min() + 1)
print(f"    ✓ ANOS_NA_PM_NORMALIZADO")

# Indicador de instituição pública
df_features['INSTITUICAO_PUBLICA'] = (df_features['Instituição de ensino'] == 'Pública').astype(int)
print(f"    ✓ INSTITUICAO_PUBLICA")

# Indicador de gênero feminino
df_features['GENERO_FEMININO'] = (df_features['Gênero'] == 'Feminino').astype(int)
print(f"    ✓ GENERO_FEMININO")

# 2.6 - Codificação de Pedra
print("\n  2.6 - Codificação de Pedra:")

# Mapear Pedra para nível (ordinal)
pedra_map = {'Quartzo': 1, 'Agata': 2, 'Ametista': 3, 'Topázio': 4}
df_features['PEDRA_NIVEL'] = df_features['PEDRA_CLASSIFICACAO'].map(pedra_map)
print(f"    ✓ PEDRA_NIVEL (1=Quartzo, 2=Ágata, 3=Ametista, 4=Topázio)")

# One-hot encoding para Pedra
pedra_dummies = pd.get_dummies(df_features['PEDRA_CLASSIFICACAO'], prefix='PEDRA')
df_features = pd.concat([df_features, pedra_dummies], axis=1)
print(f"    ✓ One-hot encoding para Pedra")

# ============================================================================
# 3. SELECIONAR FEATURES PARA MODELO
# ============================================================================
print("\n[3/6] Selecionando features para modelo...")

# Features numéricas principais (indicadores originais)
features_indicadores = ['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']

# Features derivadas
features_derivadas = [
    'MEDIA_INDICADORES', 'STD_INDICADORES', 'MIN_INDICADORES', 'MAX_INDICADORES',
    'IPP_IDA_MEDIA', 'IPP_IEG_MEDIA', 'IDA_IEG_MEDIA', 'TRIO_PRINCIPAL',
    'DISTANCIA_IAN_MEDIANA', 'BAIXO_DESEMPENHO', 'BAIXO_ENGAJAMENTO', 'BAIXO_IPP',
    'CONTAGEM_BAIXOS', 'VARIACAO_22_23', 'VARIACAO_23_24', 'TENDENCIA_GERAL',
    'DETERIORACAO_2024', 'IDADE_NORMALIZADA', 'ANOS_NA_PM_NORMALIZADO',
    'INSTITUICAO_PUBLICA', 'GENERO_FEMININO', 'PEDRA_NIVEL'
]

# Features de Pedra (one-hot)
features_pedra = [col for col in df_features.columns if col.startswith('PEDRA_') and col != 'PEDRA_CLASSIFICACAO']

# Todas as features
todas_features = features_indicadores + features_derivadas + features_pedra

print(f"\n✓ Features selecionadas:")
print(f"  - Indicadores originais: {len(features_indicadores)}")
print(f"  - Features derivadas: {len(features_derivadas)}")
print(f"  - Features de Pedra (one-hot): {len(features_pedra)}")
print(f"  - TOTAL: {len(todas_features)} features")

# ============================================================================
# 4. TRATAR VALORES FALTANTES
# ============================================================================
print("\n[4/6] Tratando valores faltantes...")

# Preencher valores faltantes em features derivadas com a média
for col in features_derivadas:
    if col in df_features.columns:
        faltantes = df_features[col].isnull().sum()
        if faltantes > 0:
            df_features[col].fillna(df_features[col].mean(), inplace=True)
            print(f"  ✓ {col}: {faltantes} valores preenchidos")

# Verificar faltantes finais
faltantes_totais = df_features[todas_features].isnull().sum().sum()
print(f"\n✓ Valores faltantes após tratamento: {faltantes_totais}")

# ============================================================================
# 5. NORMALIZAR FEATURES
# ============================================================================
print("\n[5/6] Normalizando features...")

# Criar cópia para normalização
df_normalized = df_features.copy()

# Normalizar features numéricas (exceto variáveis binárias e one-hot)
scaler = StandardScaler()

# Features para normalizar (excluir binárias)
features_para_normalizar = [
    'IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV',
    'MEDIA_INDICADORES', 'STD_INDICADORES', 'MIN_INDICADORES', 'MAX_INDICADORES',
    'IPP_IDA_MEDIA', 'IPP_IEG_MEDIA', 'IDA_IEG_MEDIA', 'TRIO_PRINCIPAL',
    'DISTANCIA_IAN_MEDIANA', 'VARIACAO_22_23', 'VARIACAO_23_24', 'TENDENCIA_GERAL',
    'IDADE_NORMALIZADA', 'ANOS_NA_PM_NORMALIZADO'
]

df_normalized[features_para_normalizar] = scaler.fit_transform(df_features[features_para_normalizar])
print(f"✓ {len(features_para_normalizar)} features normalizadas")

# ============================================================================
# 6. ANALISAR IMPORTÂNCIA DE FEATURES
# ============================================================================
print("\n[6/6] Analisando importância de features...")

# Treinar Random Forest para obter importância
X = df_features[todas_features]
y = df_features['RISCO_DEFASAGEM']

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

# Obter importância
importancia = pd.DataFrame({
    'Feature': todas_features,
    'Importancia': rf_model.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\n📊 TOP 15 FEATURES MAIS IMPORTANTES:")
print("-" * 100)
for idx, row in importancia.head(15).iterrows():
    print(f"{row['Feature']:30s} | {row['Importancia']:.4f}")

# ============================================================================
# 7. SALVAR RESULTADOS
# ============================================================================
print("\n[SALVANDO RESULTADOS]")

# Salvar dataframe com todas as features
df_features.to_csv('/home/ubuntu/dados_com_features.csv', index=False)
print(f"✓ Salvo: dados_com_features.csv ({df_features.shape[0]} x {df_features.shape[1]})")

# Salvar dataframe normalizado
df_normalized.to_csv('/home/ubuntu/dados_normalizados.csv', index=False)
print(f"✓ Salvo: dados_normalizados.csv")

# Salvar importância de features
importancia.to_csv('/home/ubuntu/importancia_features.csv', index=False)
print(f"✓ Salvo: importancia_features.csv")

# Salvar lista de features
features_dict = {
    'Indicadores': features_indicadores,
    'Derivadas': features_derivadas,
    'Pedra': features_pedra,
    'Todas': todas_features
}

with open('/home/ubuntu/features_lista.txt', 'w') as f:
    f.write("FEATURES SELECIONADAS\n")
    f.write("=" * 100 + "\n\n")
    f.write(f"INDICADORES ORIGINAIS ({len(features_indicadores)}):\n")
    for feat in features_indicadores:
        f.write(f"  - {feat}\n")
    f.write(f"\nFEATURES DERIVADAS ({len(features_derivadas)}):\n")
    for feat in features_derivadas:
        f.write(f"  - {feat}\n")
    f.write(f"\nFEATURES DE PEDRA ({len(features_pedra)}):\n")
    for feat in features_pedra:
        f.write(f"  - {feat}\n")
    f.write(f"\nTOTAL: {len(todas_features)} features\n")

print(f"✓ Salvo: features_lista.txt")

# ============================================================================
# 8. CRIAR VISUALIZAÇÃO DE IMPORTÂNCIA
# ============================================================================
print("\n[CRIANDO VISUALIZAÇÕES]")

fig, ax = plt.subplots(figsize=(12, 8))
top_features = importancia.head(15)
ax.barh(range(len(top_features)), top_features['Importancia'], color='steelblue')
ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['Feature'])
ax.set_xlabel('Importância')
ax.set_title('Top 15 Features Mais Importantes (Random Forest)', fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('/home/ubuntu/06_importancia_features.png', dpi=300, bbox_inches='tight')
print("✓ Salvo: 06_importancia_features.png")
plt.close()

# ============================================================================
# 9. RESUMO FINAL
# ============================================================================
print("\n" + "=" * 100)
print("RESUMO DO FEATURE ENGINEERING")
print("=" * 100)

print(f"""
✓ FASE 4 CONCLUÍDA COM SUCESSO!

FEATURES CRIADAS:
  - Indicadores originais: {len(features_indicadores)}
  - Features derivadas: {len(features_derivadas)}
  - Features de Pedra (one-hot): {len(features_pedra)}
  - TOTAL: {len(todas_features)} features

TOP 5 FEATURES MAIS IMPORTANTES:
""")

for idx, (_, row) in enumerate(importancia.head(5).iterrows(), 1):
    print(f"  {idx}. {row['Feature']:30s} | {row['Importancia']:.4f}")

print(f"""
ARQUIVOS GERADOS:
  ✓ dados_com_features.csv (1054 x {df_features.shape[1]})
  ✓ dados_normalizados.csv
  ✓ importancia_features.csv
  ✓ features_lista.txt
  ✓ 06_importancia_features.png

PRÓXIMA FASE:
  - Separação treino/teste
  - Treinamento de múltiplos modelos
  - Validação cruzada
  - Seleção do melhor modelo
""")

print("\n✓ Pronto para Fase 5 (Modelagem ML)!")
