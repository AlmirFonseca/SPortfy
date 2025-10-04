# -*- coding: utf-8 -*-
"""
Projeto completo de Ciência de Dados em Finanças
Ferramentas: Python, Streamlit, yfinance, Plotly, SciPy, scikit-learn

Missão: Montar um portfólio comprando 5 ativos a partir de uma oferta pré-definida de 20 ativos.

Como usar:
1) Rode localmente: `pip install -r requirements.txt` (ver lista no final) e depois `streamlit run app.py`.
2) No app, escolha:
   - 5 ativos entre os 20 pré-definidos
   - Período (1m, 3m, 6m, 1y)
   - Valor inicial (ex: 1000)
   - Taxa livre de risco (a.a.) e índice de referência (benchmark)
3) O app:
   - Coleta dados dinâmicos via yfinance
   - Limpa e prepara a base
   - Calcula métricas individuais (Retorno, Volatilidade, Beta)
   - Executa duas técnicas de otimização/análise: (i) Markowitz (Fronteira Eficiente) e (ii) Monte Carlo
   - (Opcional) Clusterização dos ativos para diversificação (KMeans)
   - Destaca carteira de Sharpe máximo e de volatilidade mínima
   - Gera gráficos (preços, correlação, fronteira) e permite baixar resultados (CSV)

Justificativa da seleção dos 20 ativos (setores, liquidez, diversificação):
- Ações brasileiras de alta liquidez e setores diversos:
  PETR4.SA (Petróleo/Gás), VALE3.SA (Mineração), ITUB4.SA (Financeiro), B3SA3.SA (Serviços Financeiros),
  ABEV3.SA (Consumo não durável), WEGE3.SA (Bens de capital), SUZB3.SA (Celulose/Papel), GGBR4.SA (Siderurgia),
  MGLU3.SA (Varejo), LREN3.SA (Varejo), VIVT3.SA (Telecom), TAEE11.SA (Energia/Transmissão),
  HAPV3.SA (Saúde), YDUQ3.SA (Educação).
  Esses papéis são conhecidos pela ampla negociação (boa liquidez histórica) e representam múltiplos setores, o que favorece a diversificação setorial.
- ETFs para exposição ampla e diferentes classes:
  BOVA11.SA (Ibovespa), SMAL11.SA (small caps), IMAB11.SA (títulos públicos atrelados ao IPCA — renda fixa), IVVB11.SA (exposição internacional ao S&P 500 em reais).
- Ações internacionais (USD) para diversificação geográfica/moeda:
  AAPL, MSFT (mega caps de tecnologia, alta liquidez). 

Observação: yfinance não expõe CDBs diretamente; usamos IMAB11.SA para representar uma classe de renda fixa local (títulos públicos longos) e ETFs para ampliação de classe/mercado.

Lógica matemática (resumo):
- Retornos diários: r_t = P_t / P_{t-1} - 1.
- Retorno esperado anualizado: E[R] = média_diária * 252.
- Volatilidade anualizada: σ = desvio_padrão_diário * sqrt(252).
- Para um vetor de pesos w (∑w=1, w≥0) e matriz de covariância Σ (diária):
    * Retorno carteira: μ_p = 252 * (w^T μ_diária)
    * Risco carteira: σ_p = sqrt(252 * w^T Σ w)
    * Sharpe (anual): S = (μ_p - r_f) / σ_p, onde r_f é taxa livre de risco anual.
- Otimização (Markowitz):
    * Máx Sharpe: maximiza S sob ∑w=1, w≥0.
    * Mín Vol: minimiza σ_p sob ∑w=1, w≥0.
- Beta (em relação ao benchmark m): β_i = Cov(R_i, R_m) / Var(R_m). Beta da carteira: β_p = ∑ w_i β_i.

"""

from __future__ import annotations
import io
import math
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from scipy.optimize import minimize
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------
# Configurações básicas do app
# ------------------------------
st.set_page_config(page_title="Portfy", layout="wide")
st.title("💰 $Portfy: Sua playlist de ativos")

st.caption("Aluno: Almir Fonseca - Professor: Diogo Robaina")

with st.expander("Sobre o projeto"):

    st.markdown(
        """
            Este projeto demonstra a aplicação de técnicas quantitativas para construção e otimização de portfólios financeiros, como entrega acadêmica da Avaliação 1 da disciplina de Ciência de Dados em Finanças da FGV EPGE.
            
            A ferramenta desenvolvida utiliza dados reais coletados via yfinance de 5 ativos simultaneamente, cálculos estatísticos e métodos de otimização clássicos (Markowitz) e estocásticos (Monte Carlo).

            **Ferramentas**: 
            > - **Python**
            > - **Streamlit** (interface web interativa e hosting)
            > - **yfinance** (coleta de dados financeiros - Yahoo Finance)
            > - **Plotly** (visualizações interativas)
            > - **SciPy** (otimização e cálculos científicos)
            > - **scikit-learn** (machine learning e análise de dados, como a clusterização por k-means)

            **Técnicas**: 
            > - **Cálculo de retornos** (diários e anualizados)
            > - **Volatilidade** (anualizada)
            > - **Beta** (em relação a um benchmark escolhido)
            > - **Otimização de portfólio** (Markowitz: Sharpe máximo e volatilidade mínima)
            > - **Simulações Monte Carlo** para análise de risco/retorno
            > - **Clusterização** para diversificação (k-means sobre embedding de correlação)

            **Objetivo**: 
            > Fornecer uma análise prática e interativa para auxiliar na tomada de decisão de investimentos.
        """
    )
    st.divider()
    
    st.warning("**NOTA**: Este é um projeto acadêmico e não constitui recomendação financeira.")

# ------------------------------
# Universo de 20 ativos
# ------------------------------

UNIVERSE = [
    "PETR4.SA",  # Petrobras — Petróleo e gás (energia)
    "VALE3.SA",  # Vale — Mineração e metais (commodities)
    "ITUB4.SA",  # Itaú Unibanco — Bancos e serviços financeiros (financeiro)
    "B3SA3.SA",  # B3 (Brasil, Bolsa, Balcão) — Infraestrutura de mercado (financeiro)
    "ABEV3.SA",  # Ambev — Bebidas e consumo não durável (consumo)
    "WEGE3.SA",  # WEG — Bens de capital e equipamentos elétricos (industrial)
    "SUZB3.SA",  # Suzano — Celulose e papel (commodities)
    "GGBR4.SA",  # Gerdau — Siderurgia e metalurgia (commodities)
    "MGLU3.SA",  # Magazine Luiza — Varejo e e-commerce (consumo)
    "LREN3.SA",  # Lojas Renner — Varejo de moda (consumo)
    "VIVT3.SA",  # Telefônica Brasil (Vivo) — Telecomunicações
    "TAEE11.SA", # Taesa — Transmissão de energia elétrica (utilidade pública)
    "HAPV3.SA",  # Hapvida — Saúde e planos médicos (serviços)
    "YDUQ3.SA",  # Yduqs — Educação privada (serviços)
    "BOVA11.SA", # iShares Ibovespa — ETF de mercado amplo (broad market BR)
    "SMAL11.SA", # iShares Small Caps — ETF de small caps brasileiras
    "IMAB11.SA", # iShares IMA-B — ETF de títulos públicos indexados ao IPCA (renda fixa)
    "IVVB11.SA", # iShares S&P 500 — ETF que replica o S&P 500 em reais (exposição internacional)
    "BBAS3.SA",  # Banco do Brasil — Bancos e serviços financeiros estatais (financeiro)
    "EQTL3.SA"   # Equatorial Energia — Distribuição e transmissão de energia (utilidade pública)
]


# ------------------------------
# Funções utilitárias de dados
# ------------------------------

def fetch_prices(tickers: list[str], period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Coleta preços via yfinance e retorna base limpa (preferindo 'Adj Close' se disponível).
    - Remove colunas totalmente vazias
    - Forward-fill e back-fill para pequenos buracos
    - Drop de linhas onde tudo continua NaN
    """
    data = yf.download(tickers, period=period, interval=interval, auto_adjust=False, progress=False)
    
    # Verifica se há MultiIndex (vários tickers)
    if isinstance(data.columns, pd.MultiIndex):
        # Usa 'Adj Close' se existir; caso contrário, usa 'Close'
        if 'Adj Close' in data.columns.get_level_values(0):
            prices = data['Adj Close'].copy()
        elif 'Close' in data.columns.get_level_values(0):
            prices = data['Close'].copy()
        else:
            raise KeyError("Nem 'Adj Close' nem 'Close' foram encontrados nos dados.")
    else:
        # Caso de 1 ticker
        if 'Adj Close' in data.columns:
            prices = data[['Adj Close']].copy()
        elif 'Close' in data.columns:
            prices = data[['Close']].copy()
        else:
            raise KeyError("Nem 'Adj Close' nem 'Close' foram encontrados nos dados.")

    # Limpeza
    prices = prices.dropna(axis=1, how="all").ffill().bfill().dropna(how="all")

    # Renomeia colunas se necessário (para 1 ticker, deixa nome claro)
    if len(prices.columns) == 1 and not isinstance(tickers, str):
        prices.columns = [tickers[0]]

    return prices

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="all")
    return returns


def annualize_stats(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    mu_daily = returns.mean()
    cov_daily = returns.cov()
    mu_annual = mu_daily * 252
    cov_annual = cov_daily * 252
    return mu_annual, cov_annual


def portfolio_perf(weights: np.ndarray, mu_annual: pd.Series, cov_annual: pd.DataFrame, rf: float) -> tuple[float, float, float]:
    weights = np.array(weights)
    mu_p = float(np.dot(weights, mu_annual.values))
    vol_p = float(np.sqrt(np.dot(weights.T, np.dot(cov_annual.values, weights))))
    sharpe = (mu_p - rf) / vol_p if vol_p > 0 else -np.inf
    return mu_p, vol_p, sharpe


def max_sharpe(mu_annual, cov_annual, rf, bounds, x0=None):
    n = len(mu_annual)
    if x0 is None:
        x0 = np.ones(n) / n

    def neg_sharpe(w):
        return -portfolio_perf(w, mu_annual, cov_annual, rf)[2]

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    res = minimize(neg_sharpe, x0=x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000})
    return res


def min_vol(mu_annual, cov_annual, bounds, x0=None):
    n = len(mu_annual)
    if x0 is None:
        x0 = np.ones(n) / n

    def vol(w):
        return portfolio_perf(w, mu_annual, cov_annual, 0.0)[1]

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    res = minimize(vol, x0=x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000})
    return res


def simulate_monte_carlo(mu_annual, cov_annual, rf, n_portfolios=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(mu_annual)
    ws, mus, vols, sharpes = [], [], [], []
    for _ in range(n_portfolios):
        w = rng.random(n)
        w = w / w.sum()
        mu_p, vol_p, s = portfolio_perf(w, mu_annual, cov_annual, rf)
        ws.append(w)
        mus.append(mu_p)
        vols.append(vol_p)
        sharpes.append(s)
    df = pd.DataFrame(ws, columns=mu_annual.index)
    df["ret"], df["vol"], df["sharpe"] = mus, vols, sharpes
    return df


def compute_beta(returns: pd.DataFrame, benchmark: pd.Series) -> pd.Series:
    # beta_i = Cov(Ri, Rm) / Var(Rm)
    betas = {}
    var_m = benchmark.var()
    for col in returns.columns:
        cov_im = returns[col].cov(benchmark)
        betas[col] = cov_im / var_m if var_m > 0 else np.nan
    return pd.Series(betas)


def kmeans_clusters(returns: pd.DataFrame, k: int = 3, seed: int = 42) -> pd.Series:
    # usa matriz de correlação para derivar distância (1 - corr)
    corr = returns.corr().fillna(0)
    dist = 1 - corr
    # embedding simples: usa as duas primeiras componentes da MDS aproximada (PCA da dist centrada)
    # para simplificar, vamos usar os autovetores principais da matriz de correlação
    vals, vecs = np.linalg.eigh(corr.values)
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx[:2]]
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(vecs)
    return pd.Series(labels, index=corr.index)

# ------------------------------
# Sidebar — parâmetros do usuário
# ------------------------------
with st.sidebar:
    st.header("Parâmetros do Estudo")

    chosen = st.multiselect("Escolha 5 ativos", UNIVERSE, default=["BOVA11.SA", "ITUB4.SA", "VALE3.SA", "WEGE3.SA", "IVVB11.SA"])
    if len(chosen) != 5:
        st.warning("⚠️ Por favor, selecione **exatamente 5** ativos.")

    period_label = st.segmented_control("Período histórico", ["1m", "3m", "6m", "1a", "5a"], default="6m", width="stretch")
    period_map = {"1m": "1mo", "3m": "3mo", "6m": "6mo", "1a": "1y", "5a": "5y"}
    period = period_map[period_label]

    initial_capital = st.number_input("Valor inicial (R$)", min_value=100.0, value=1000.0, step=50.0, format="%.2f")
    
    rf = st.slider("Taxa livre de risco anual (ex: 0.10 = 10%)", min_value=0.0, max_value=0.20, value=0.10, step=0.001, format="%.3f")
    
    benchmark_options = {
        "^BVSP (Ibovespa)": "^BVSP", 
        "BOVA11.SA (Ibovespa ETF)": "BOVA11.SA", 
        "^GSPC (S&P 500)": "^GSPC", 
        "^IBX50 (IBrX 50)": "^IBX50", 
        "BRAX11.SA (IBrX 100)": "BRAX11.SA"
    }

    benchmark_opt = st.selectbox("Benchmark", list(benchmark_options.keys()), index=0)
    benchmark_ticker = benchmark_options[benchmark_opt]

# Se a seleção de 5 não está correta, não segue
if len(chosen) != 5:
    st.stop()

# ------------------------------
# Coleta de dados (dinâmica) e limpeza
# ------------------------------
st.subheader("1) Coleta e Limpeza de Dados")

with st.spinner("Baixando preços via yfinance..."):
    prices_sel = fetch_prices(chosen, period=period, interval="1d")
    prices_bench = fetch_prices([benchmark_ticker], period=period, interval="1d")
    
    
# Base limpa
returns_sel = compute_returns(prices_sel)
returns_bench = compute_returns(prices_bench).iloc[:, 0]

# --- Criar as abas ---
tab1_1, tab1_2, tab1_3 = st.tabs(["💎 Resumo", "📈 Base bruta (preços ajustados)", "🧹 Base limpa (retornos diários)"])

    
with tab1_1:
    metric_columns = st.columns(5)
    
    # Print the value, delta and chart for each of the 5 tickers:
    # st.metric(label="Temperature", value="70 °F", delta="1.2 °F")
    for ticker_index in range(len(chosen)):
        ticker = chosen[ticker_index]
        ret_total = (prices_sel[ticker].iloc[-1] / prices_sel[ticker].iloc[0]) - 1
        
        metric_columns[ticker_index].metric(label=ticker, value=f"{prices_sel[ticker].iloc[-1]:,.2f}", delta=f"{ret_total:.2%}", chart_data=prices_sel[ticker].tolist(), chart_type="line", border=True)

with tab1_2:
    st.dataframe(prices_sel.style.format("R$ {:,.2f}"), height=200)
    
with tab1_3:
    st.dataframe(returns_sel.style.format("{:.2%}"), height=200)
    
st.divider()

# ------------------------------
# Métricas individuais
# ------------------------------
st.subheader("2) Métricas Individuais")
mu_annual, cov_annual = annualize_stats(returns_sel)
vol_annual = np.sqrt(np.diag(cov_annual))

betas = compute_beta(returns_sel, returns_bench)

metrics_df = pd.DataFrame({
    "Retorno esperado (a.a.)": mu_annual,
    "Volatilidade (a.a.)": vol_annual,
    "Beta": betas
}).loc[chosen]

st.dataframe(metrics_df.style.format({"Retorno esperado (a.a.)": "{:.2%}", "Volatilidade (a.a.)": "{:.2%}", "Beta": "{:.2f}"}))

# ------------------------------
# Visualizações: preços e correlações
# ------------------------------
st.subheader("3) Visualizações: Preços e Correlação")

# --- Criar as abas ---
tab3_1, tab3_2 = st.tabs(["📈 Preços Ajustados", "⚖️ Preços Normalizados (Base 100)"])

# --- Aba 1 ---
with tab3_1:
    px_prices = px.line(prices_sel, labels={"value": "Preço (R$)", "Date": "Data"})
    st.plotly_chart(px_prices, use_container_width=True)

# --- Aba 2 ---
with tab3_2:
    prices_norm = prices_sel / prices_sel.iloc[0] * 100
    px_norm = px.line(prices_norm, labels={"value": "Índice (Base 100)", "Date": "Data"})
    st.plotly_chart(px_norm, use_container_width=True)

corr = returns_sel.corr()
px_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de Correlação dos Retornos")
st.plotly_chart(px_corr, use_container_width=True)

# Clusters (opcional para diversificação)
try:
    clusters = kmeans_clusters(returns_sel, k=min(3, len(chosen)))
    clust_df = clusters.rename("Cluster").to_frame()
    st.write("**Clusters (KMeans sobre embedding de correlação):**", clust_df)
except Exception as e:
    st.info(f"Clusterização não disponível: {e}")

# ------------------------------
# Otimização: Markowitz + Monte Carlo
# ------------------------------
st.subheader("4) Otimização de Portfólio (Markowitz + Monte Carlo)")

bounds = tuple((0.0, 1.0) for _ in chosen)

# Otimização por Markowitz
res_ms = max_sharpe(mu_annual.loc[chosen], cov_annual.loc[chosen, chosen], rf, bounds)
res_mv = min_vol(mu_annual.loc[chosen], cov_annual.loc[chosen, chosen], bounds)

w_ms = res_ms.x / res_ms.x.sum()
w_mv = res_mv.x / res_mv.x.sum()

mu_ms, vol_ms, s_ms = portfolio_perf(w_ms, mu_annual.loc[chosen], cov_annual.loc[chosen, chosen], rf)
mu_mv, vol_mv, s_mv = portfolio_perf(w_mv, mu_annual.loc[chosen], cov_annual.loc[chosen, chosen], rf)

# Monte Carlo
mc_df = simulate_monte_carlo(mu_annual.loc[chosen], cov_annual.loc[chosen, chosen], rf, n_portfolios=4000)

# Fronteira aproximada (varrer alvos de retorno)

def efficient_frontier(mu, cov, rf, n_points=40):
    """Traça uma aproximação da fronteira eficiente resolvendo mín vol para retornos alvo."""
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    cons_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    targets = np.linspace(mu.min(), mu.max(), n_points)
    vols, rets, ws = [], [], []

    for t in targets:
        cons_ret = {'type': 'eq', 'fun': lambda w, t=t: np.dot(w, mu.values) - t}
        def vol(w):
            return math.sqrt(np.dot(w.T, np.dot(cov.values, w)))
        res = minimize(vol, x0=np.ones(n)/n, method='SLSQP', bounds=bounds, constraints=(cons_sum, cons_ret))
        if res.success:
            w = res.x / res.x.sum()
            ws.append(w)
            rets.append(t)
            vols.append(vol(w))
    ef = pd.DataFrame({"vol": vols, "ret": rets})
    return ef

ef = efficient_frontier(mu_annual.loc[chosen], cov_annual.loc[chosen, chosen], rf)

# Scatter com Monte Carlo + destaques e fronteira
fig_sc = px.scatter(mc_df, x="vol", y="ret", opacity=0.2, labels={"vol": "Volatilidade (a.a.)", "ret": "Retorno (a.a.)"}, title="Fronteira Eficiente e Simulações (Monte Carlo)")
fig_sc.add_trace(go.Scatter(x=ef["vol"], y=ef["ret"], mode="lines", name="Fronteira Eficiente"))
fig_sc.add_trace(go.Scatter(x=[vol_ms], y=[mu_ms], mode="markers", marker_symbol="star", marker_size=14, name="Sharpe Máximo"))
fig_sc.add_trace(go.Scatter(x=[vol_mv], y=[mu_mv], mode="markers", marker_symbol="diamond", marker_size=12, name="Volatilidade Mínima"))
st.plotly_chart(fig_sc, use_container_width=True)

# Tabelas de pesos ideais
weights_df = pd.DataFrame({
    "Ativo": chosen,
    "Peso - Sharpe Máximo": w_ms,
    "Peso - Vol Mín": w_mv
}).set_index("Ativo")

st.dataframe(weights_df.style.format({"Peso - Sharpe Máximo": "{:.2%}", "Peso - Vol Mín": "{:.2%}"}))

# Projeção com valor inicial
alloc_sharpe = (w_ms * initial_capital)
alloc_minvol = (w_mv * initial_capital)

alloc_df = pd.DataFrame({
    "Ativo": chosen,
    "Alocação (R$) - Sharpe Máximo": alloc_sharpe,
    "Alocação (R$) - Vol Mín": alloc_minvol
}).set_index("Ativo")

st.dataframe(alloc_df.style.format({"Alocação (R$) - Sharpe Máximo": "R$ {:,.2f}", "Alocação (R$) - Vol Mín": "R$ {:,.2f}"}))

pie_alloc = go.Figure()
pie_alloc.add_trace(go.Pie(labels=chosen, values=w_ms, hole=0.6, title="Alocação — Sharpe Máximo"))
st.plotly_chart(pie_alloc, use_container_width=True)

# ------------------------------
# Comparação com Benchmark e Beta de carteira
# ------------------------------

st.subheader("5) Comparação com Benchmark e Beta da Carteira")


ret_bench_total = (prices_bench[benchmark_ticker].iloc[-1] / prices_bench[benchmark_ticker].iloc[0]) - 1
st.metric(label=benchmark_ticker, value=f"{prices_bench[benchmark_ticker].iloc[-1]:,.2f}", delta=f"{ret_bench_total:.2%}", chart_data=prices_bench[benchmark_ticker].tolist(), chart_type="line", border=True)


# Construir série de valor do portfólio (Sharpe Máx) ao longo do tempo
w_ms_series = pd.Series(w_ms, index=chosen)
ret_port_daily = (returns_sel[chosen] * w_ms_series).sum(axis=1)
bench_daily = returns_bench.reindex(ret_port_daily.index).fillna(0)

cum_port = (1 + ret_port_daily).cumprod()
cum_bench = (1 + bench_daily).cumprod()

comp_df = pd.DataFrame({"Portfólio (Sharpe Máx)": cum_port, "Benchmark": cum_bench})
comp_fig = px.line(comp_df, title="Evolução do Valor — Portfólio vs Benchmark", labels={"value": "Índice acumulado", "index": "Data"})
st.plotly_chart(comp_fig, use_container_width=True)

beta_assets = betas.loc[chosen]
beta_port = float((beta_assets * w_ms_series).sum())

benchmark_df = pd.DataFrame({
    "Beta Ativo": beta_assets, 
    "Peso": w_ms_series, 
    "Contribuição Beta": beta_assets * w_ms_series
})

st.dataframe(benchmark_df.style.format({"Beta Ativo": "{:.4f}", "Peso": "{:.3%}", "Contribuição Beta": "{:.4f}"}))

st.info(f"**Beta da carteira (Sharpe Máx)**: {beta_port:.2f}")

# ------------------------------
# Download dos resultados
# ------------------------------
st.subheader("6) Exportar")

raw_bytes = prices_sel.to_csv(index=True).encode("utf-8")
st.download_button("Baixar dados brutos (CSV)", data=raw_bytes, file_name="precos_brutos.csv", mime="text/csv", width="stretch")

out_stats = metrics_df.copy()
out_stats["Peso Sharpe Máx"] = w_ms_series
out_stats["Peso Vol Mín"] = pd.Series(w_mv, index=chosen)

csv_bytes = out_stats.to_csv(index=True).encode("utf-8")
st.download_button("Baixar métricas e pesos (CSV)", data=csv_bytes, file_name="metricas_pesos.csv", mime="text/csv", width="stretch")


# ToDo:
# - Listar todas as empresas disponíveis na B3, com nome e setor
# - Permitir escolher mais de 5 ativos (até 10?) e limitar a 5 na otimização
# - Permitir escolher ativos internacionais (USD) e converter para BRL via câmbio (ex: AAPL, MSFT), atualizado conforme o câmbio histórico
#  - Adicionar métricas como Drawdown
# - Permitir escolher mais de 1 benchmark (ex: BOVA11 + IVVB11) e calcular beta composto
# - Adicionar ajudas (help) explicando cada métrica e conceito
# - Melhorar visualizações (ex: adicionar médias móveis, bandas de Bollinger, candlesticks, etc)
# - Adicionar formatação condicional nos dataframes (ex: máximos, mínimos, intervalos contínuos, etc)