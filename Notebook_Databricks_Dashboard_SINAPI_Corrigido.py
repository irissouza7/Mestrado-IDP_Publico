# Databricks notebook source
# MAGIC %md
# MAGIC # Dashboard SINAPI: O CEP da obra muda o orçamento?
# MAGIC Notebook corrigido, sem marcadores de Markdown dentro das células Python.

# COMMAND ----------

# MAGIC %pip install plotly pandas numpy

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
from html import escape

CATALOG = ""
SCHEMA = ""
DATA_INICIAL = "2024-04-01"
DATA_FINAL = "2026-07-01"
REGIME_PRINCIPAL = "padrao"
TOP_UFS = 15
TOP_COMPOSICOES = 15
ARQUIVO_HTML_LOCAL = "/tmp/dashboard_sinapi.html"
ARQUIVO_HTML_DBFS = "dbfs:/FileStore/dashboard_sinapi.html"
LINK_FILESTORE = "/files/dashboard_sinapi.html"

# COMMAND ----------

def nome_completo_tabela(nome_tabela):
    partes = [parte for parte in [CATALOG, SCHEMA, nome_tabela] if parte]
    return ".".join(partes)

TABELAS = {
    "custos": nome_completo_tabela("sinapi_RT_relatorios_custos"),
    "uf_encargos": nome_completo_tabela("sinapi_RT_UF_encargos"),
    "versoes": nome_completo_tabela("sinapi_Vrts_historicas_vigentes"),
    "coeficientes": nome_completo_tabela("RT_Coeficientes"),
    "composicoes": nome_completo_tabela("sinapi_RT_Composicoes"),
    "insumos": nome_completo_tabela("sinapi_RT_Insumos"),
    "precos_gepad": nome_completo_tabela("sinapi_RT_Precos_GEPAD"),
    "precos_ibge": nome_completo_tabela("sinapi_RT_Precos_IBGE"),
    "analitico": nome_completo_tabela("sinapi_RT_Analitico")
}

for apelido, tabela in TABELAS.items():
    print(f"{apelido:15s} -> {tabela}")

# COMMAND ----------

def carregar_tabela(nome):
    print(f"Carregando: {nome}")
    return spark.table(nome)

df_custos = carregar_tabela(TABELAS["custos"])
df_uf_encargos = carregar_tabela(TABELAS["uf_encargos"])
df_versoes = carregar_tabela(TABELAS["versoes"])
df_coeficientes = carregar_tabela(TABELAS["coeficientes"])
df_composicoes = carregar_tabela(TABELAS["composicoes"])
df_insumos = carregar_tabela(TABELAS["insumos"])
df_precos_gepad = carregar_tabela(TABELAS["precos_gepad"])
df_precos_ibge = carregar_tabela(TABELAS["precos_ibge"])
df_analitico = carregar_tabela(TABELAS["analitico"])

# COMMAND ----------

# Inventário das variáveis
dataframes = {
    "sinapi_RT_relatorios_custos": df_custos,
    "sinapi_RT_UF_encargos": df_uf_encargos,
    "sinapi_Vrts_historicas_vigentes": df_versoes,
    "RT_Coeficientes": df_coeficientes,
    "sinapi_RT_Composicoes": df_composicoes,
    "sinapi_RT_Insumos": df_insumos,
    "sinapi_RT_Precos_GEPAD": df_precos_gepad,
    "sinapi_RT_Precos_IBGE": df_precos_ibge,
    "sinapi_RT_Analitico": df_analitico
}

inventario = []
for nome, dataframe in dataframes.items():
    quantidade = dataframe.count()
    for campo in dataframe.schema.fields:
        inventario.append({
            "tabela": nome,
            "variavel": campo.name,
            "tipo": campo.dataType.simpleString(),
            "aceita_nulo": campo.nullable,
            "quantidade_registros_tabela": quantidade
        })

df_inventario = spark.createDataFrame(inventario)
display(df_inventario)

# COMMAND ----------

# Dimensões
dim_tempo = (
    df_versoes
    .select(
        F.col("rt_id").cast("long").alias("rt_id"),
        F.to_date("data_coleta").alias("data_coleta"),
        F.to_date("data_rt").alias("data_rt"),
        F.col("versao").cast("long").alias("versao"),
        F.col("notas_da_versao")
    )
    .filter(F.col("data_rt").between(F.to_date(F.lit(DATA_INICIAL)), F.to_date(F.lit(DATA_FINAL))))
    .dropDuplicates(["rt_id"])
)

dim_uf = (
    df_uf_encargos
    .select(
        F.col("rt_id").cast("long").alias("rt_id_uf"),
        F.col("ugi").cast("long").alias("ugi_uf"),
        F.upper(F.trim("uf")).alias("uf"),
        F.trim("localidade").alias("localidade"),
        F.col("encargo_padrao_horista").cast("double"),
        F.col("encargo_padrao_mensalista").cast("double"),
        F.col("encargo_desonerado_horista").cast("double"),
        F.col("encargo_desonerado_mensalista").cast("double")
    )
    .filter(F.col("uf").isNotNull())
    .dropDuplicates(["rt_id_uf", "ugi_uf", "uf"])
)

dim_composicao = (
    df_composicoes
    .select(
        F.col("rt_id").cast("long").alias("rt_id_comp"),
        F.col("codigo_composicao").cast("long").alias("codigo_composicao"),
        F.trim("descricao").alias("descricao"),
        F.trim("unidade").alias("unidade"),
        F.trim("situacao").alias("situacao"),
        F.trim("grupo").alias("grupo")
    )
    .dropDuplicates(["rt_id_comp", "codigo_composicao"])
)

# COMMAND ----------

# Base analítica
custos_composicoes = (
    df_custos
    .select(
        F.col("rt_id").cast("long").alias("rt_id"),
        F.col("ugi").cast("long").alias("ugi"),
        F.upper(F.trim("tipo")).alias("tipo"),
        F.col("codigo").cast("long").alias("codigo"),
        F.trim("origem_preco").alias("origem_preco"),
        F.col("padrao").cast("double").alias("padrao"),
        F.col("desonerado").cast("double").alias("desonerado"),
        F.col("sem_encargos").cast("double").alias("sem_encargos")
    )
    .filter(F.col("tipo").contains("COMPOS"))
)

base_analitica = (
    custos_composicoes.alias("c")
    .join(dim_tempo.alias("t"), F.col("c.rt_id") == F.col("t.rt_id"), "inner")
    .join(
        dim_uf.alias("u"),
        (F.col("c.rt_id") == F.col("u.rt_id_uf")) & (F.col("c.ugi") == F.col("u.ugi_uf")),
        "left"
    )
    .join(
        dim_composicao.alias("d"),
        (F.col("c.rt_id") == F.col("d.rt_id_comp")) &
        (F.col("c.codigo") == F.col("d.codigo_composicao")),
        "left"
    )
    .select(
        F.col("c.rt_id"), F.col("c.ugi"), F.col("t.data_rt"),
        F.date_format(F.col("t.data_rt"), "yyyy-MM").alias("competencia"),
        F.col("t.versao"), F.col("u.uf"), F.col("u.localidade"),
        F.col("c.codigo").alias("codigo_composicao"),
        F.col("d.descricao"), F.col("d.unidade"), F.col("d.grupo"), F.col("d.situacao"),
        F.col("c.origem_preco"), F.col("c.padrao"), F.col("c.desonerado"), F.col("c.sem_encargos")
    )
    .withColumn(
        "valor",
        F.when(F.lit(REGIME_PRINCIPAL) == "padrao", F.col("padrao"))
         .when(F.lit(REGIME_PRINCIPAL) == "desonerado", F.col("desonerado"))
         .otherwise(F.col("sem_encargos"))
    )
    .filter(
        F.col("uf").isNotNull() & F.col("descricao").isNotNull() &
        F.col("valor").isNotNull() & (F.col("valor") > 0)
    )
    .cache()
)

print(f"Registros da base analítica: {base_analitica.count():,}")
display(base_analitica.limit(20))

# COMMAND ----------

# Indicadores
indicadores = base_analitica.agg(
    F.count("*").alias("registros"),
    F.countDistinct("competencia").alias("meses"),
    F.countDistinct("uf").alias("ufs"),
    F.countDistinct("codigo_composicao").alias("composicoes"),
    F.min("data_rt").alias("data_minima"),
    F.max("data_rt").alias("data_maxima"),
    F.expr("percentile_approx(valor, 0.5)").alias("custo_mediano")
).first()

print(indicadores)

# COMMAND ----------

# Evolução temporal
serie_item = base_analitica.groupBy(
    "data_rt", "competencia", "uf", "codigo_composicao"
).agg(F.expr("percentile_approx(valor, 0.5)").alias("valor_item"))

janela_item = Window.partitionBy("uf", "codigo_composicao").orderBy("data_rt")
serie_item = (
    serie_item
    .withColumn("valor_anterior", F.lag("valor_item").over(janela_item))
    .withColumn(
        "variacao_mensal_pct",
        F.when(F.col("valor_anterior") > 0, (F.col("valor_item") / F.col("valor_anterior") - 1) * 100)
    )
)

evolucao_mensal = serie_item.groupBy("data_rt", "competencia").agg(
    F.expr("percentile_approx(variacao_mensal_pct, 0.5)").alias("variacao_mediana_pct")
).orderBy("data_rt")
pdf_evolucao = evolucao_mensal.toPandas()

# COMMAND ----------

# Ranking territorial
cobertura_composicao_uf = base_analitica.groupBy("uf", "codigo_composicao").agg(
    F.countDistinct("competencia").alias("meses_observados"),
    F.expr("percentile_approx(valor, 0.5)").alias("valor_mediano_item")
).filter(F.col("meses_observados") >= 6)

ranking_uf = cobertura_composicao_uf.groupBy("uf").agg(
    F.expr("percentile_approx(valor_mediano_item, 0.5)").alias("custo_mediano_uf"),
    F.countDistinct("codigo_composicao").alias("composicoes_comparaveis")
).filter(F.col("composicoes_comparaveis") >= 30).orderBy(F.desc("custo_mediano_uf"))

pdf_ranking_uf = ranking_uf.toPandas()

# COMMAND ----------

# Gráficos
COR_AZUL = "#005CA9"
COR_LARANJA = "#F39200"
COR_CINZA = "#5F6B76"

fig_evolucao = go.Figure()
fig_evolucao.add_trace(go.Scatter(
    x=pdf_evolucao["data_rt"],
    y=pdf_evolucao["variacao_mediana_pct"],
    mode="lines+markers",
    line=dict(color=COR_AZUL, width=3),
    marker=dict(size=7, color=COR_LARANJA),
    hovertemplate="<b>%{x|%m/%Y}</b><br>Variação mediana: %{y:.2f}%<extra></extra>"
))
fig_evolucao.add_hline(y=0, line_dash="dash", line_color=COR_CINZA)
fig_evolucao.update_layout(
    title="A variação mediana revela os meses de maior movimentação dos custos",
    xaxis_title="Competência",
    yaxis_title="Variação mediana mensal (%)",
    template="plotly_white",
    height=430,
    showlegend=False
)

ranking_plot = pdf_ranking_uf.head(TOP_UFS).sort_values("custo_mediano_uf", ascending=True)
fig_uf = go.Figure(go.Bar(
    x=ranking_plot["custo_mediano_uf"],
    y=ranking_plot["uf"],
    orientation="h",
    marker_color=COR_AZUL,
    customdata=ranking_plot[["composicoes_comparaveis"]].to_numpy(),
    hovertemplate=(
        "<b>%{y}</b><br>Mediana: R$ %{x:,.2f}<br>"
        "Composições comparáveis: %{customdata[0]:,.0f}<extra></extra>"
    )
))
fig_uf.update_layout(
    title="Mesmo com itens comparáveis, os custos não se distribuem igualmente pelo território",
    xaxis_title="Custo mediano das composições comparáveis (R$)",
    yaxis_title="UF",
    template="plotly_white",
    height=520
)

# COMMAND ----------

# Dashboard HTML corrigido

def figura_para_html(figura):
    return pio.to_html(
        figura,
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True}
    )

html_evolucao = figura_para_html(fig_evolucao)
html_uf = figura_para_html(fig_uf)

data_minima_txt = indicadores["data_minima"].strftime("%d/%m/%Y")
data_maxima_txt = indicadores["data_maxima"].strftime("%d/%m/%Y")

uf_maior = pdf_ranking_uf.iloc[0]["uf"] if not pdf_ranking_uf.empty else "N/D"
valor_uf_maior = pdf_ranking_uf.iloc[0]["custo_mediano_uf"] if not pdf_ranking_uf.empty else np.nan

def numero_br(valor, casas=0):
    if valor is None or pd.isna(valor):
        return "N/D"
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")

def moeda_br(valor):
    return f"R$ {numero_br(valor, 2)}"

dashboard_html = f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>O CEP da obra muda o orçamento?</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
body {{ margin: 0; background: #F4F7FA; color: #1F2937; font-family: Arial, sans-serif; }}
.hero {{ padding: 55px 7%; color: white; background: linear-gradient(120deg, #003B70, #005CA9, #0086BF); }}
.hero h1 {{ margin: 0; font-size: 46px; }}
.pergunta {{ margin-top: 24px; padding: 18px; border-left: 6px solid #F39200; background: rgba(255,255,255,0.12); font-size: 23px; font-weight: bold; }}
.container {{ max-width: 1300px; margin: auto; padding: 30px 5% 60px; }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }}
.card, .secao {{ background: white; border-radius: 16px; padding: 22px; box-shadow: 0 8px 24px rgba(31,41,55,0.08); }}
.card-valor {{ color: #005CA9; font-size: 29px; font-weight: bold; }}
.secao {{ margin-top: 28px; }}
.secao h2 {{ color: #005CA9; }}
.fonte {{ margin-top: 28px; padding: 22px; background: #EAF4FB; border-radius: 14px; font-size: 14px; }}
@media (max-width: 850px) {{ .cards {{ grid-template-columns: 1fr 1fr; }} }}
</style>
</head>
<body>
<header class="hero">
<h1>O CEP da obra muda o orçamento?</h1>
<p>Uma jornada pelos custos do SINAPI entre abril de 2024 e julho de 2026.</p>
<div class="pergunta">Quanto o local e o tempo influenciam o custo de construir no Brasil?</div>
</header>
<main class="container">
<section class="cards">
<div class="card"><div class="card-valor">{numero_br(indicadores['registros'])}</div><div>registros analisados</div></div>
<div class="card"><div class="card-valor">{numero_br(indicadores['meses'])}</div><div>competências mensais</div></div>
<div class="card"><div class="card-valor">{numero_br(indicadores['ufs'])}</div><div>unidades da Federação</div></div>
<div class="card"><div class="card-valor">{numero_br(indicadores['composicoes'])}</div><div>composições distintas</div></div>
</section>
<section class="secao">
<h2>Ato 1: o custo muda ao longo do tempo</h2>
<p>Período efetivamente analisado: {data_minima_txt} a {data_maxima_txt}.</p>
{html_evolucao}
</section>
<section class="secao">
<h2>Ato 2: o território importa</h2>
<p>Entre as UFs que atenderam aos critérios de cobertura, {escape(str(uf_maior))} apresentou o maior custo mediano comparável: {moeda_br(valor_uf_maior)}.</p>
{html_uf}
</section>
<section class="secao">
<h2>Conclusão</h2>
<p>A referência é nacional, mas o custo é territorial e temporal. Diferenças e variações funcionam como sinais para investigação e não comprovam irregularidade.</p>
</section>
<section class="fonte">
<strong>Fonte:</strong> SINAPI, CAIXA Econômica Federal e IBGE. Base histórica disponível no Databricks, relacionada principalmente por rt_id.
</section>
</main>
</body>
</html>
"""

# COMMAND ----------

# Salvar e visualizar
with open(ARQUIVO_HTML_LOCAL, "w", encoding="utf-8") as arquivo:
    arquivo.write(dashboard_html)

dbutils.fs.put(ARQUIVO_HTML_DBFS, dashboard_html, overwrite=True)
print(f"Arquivo salvo em: {ARQUIVO_HTML_DBFS}")
print(f"Caminho provável de acesso: {LINK_FILESTORE}")
displayHTML(dashboard_html)
