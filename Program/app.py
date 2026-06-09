import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pathlib import Path

# ------------------------------------------------------
#  PAGE CONFIG & THEME
# ------------------------------------------------------

st.set_page_config(
    page_title="GPU Market Analytics",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ...existing code...
CUSTOM_CSS = """
<style>
  /* Общий фон и основной текст */
  html, body, .stApp, .reportview-container, .main {
    background-color: #0b0c0f !important;
    color: #ffffff !important;
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial !important;
  }

  /* Заголовки и важный текст — увеличить контраст */
  h1, h2, h3, h4, h5, h6, .stHeader, .stMarkdown, .css-1v0mbdj, .css-1mk3d6x {
    color: #ffffff !important;
    font-weight: 600 !important;
    text-shadow: none !important;
    opacity: 1 !important;
  }

  /* Параграфы и обычный текст */
  p, span, .stText, .stMarkdown p, .stMarkdown span {
    color: #f5f6f7 !important;
    opacity: 1 !important;
    font-weight: 500 !important;
  }

  /* Сайдбар */
  [data-testid="stSidebar"] {
    background-color: #0f1116 !important;
    color: #ffffff !important;
  }
  [data-testid="stSidebar"] * {
    color: #f5f6f7 !important;
  }

  /* Стили вкладок */
  .stTabs [role="tab"] {
    background-color: #15171c !important;
    color: #e6eef8 !important;
    padding: 8px 18px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
  }
  .stTabs [role="tab"][aria-selected="true"] {
    background-color: #313542 !important;
    color: #ffffff !important;
  }

  /* Метрики, кнопки и элементы управления */
  .stMetric, .stMetricValue, .stButton button, .stSelectbox, .stSlider {
    color: #ffffff !important;
  }
  .stButton button {
    background-color: #22252b !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
  }

  /* Таблицы и датафреймы */
  .stDataFrame, .element-container, .stAgGrid {
    color: #f5f6f7 !important;
    background-color: transparent !important;
  }
  .stDataFrame table td, .stDataFrame table th {
    color: #e9eef6 !important;
  }

  /* Ссылки */
  a, a:link, a:visited {
    color: #9ad1ff !important;
  }

  /* Удаляем затемнения у некоторых компонентов */
  [style*="opacity: 0.6"], [style*="opacity:0.6"] {
    opacity: 1 !important;
  }

  /* Дополнительно: повысить читаемость подсказок/подписи */
  .css-1q8dd3e, .css-1d391kg {
    color: #eef3f8 !important;
  }
</style>
"""
# ...existing code...
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_DIR = Path("data")

# ------------------------------------------------------
#  HELPER FUNCTIONS
# ------------------------------------------------------

def infer_vendor_from_name(name: str) -> str:
    """
    Автоматичне визначення вендора (NVIDIA / AMD / INTEL / OTHER)
    """
    if not isinstance(name, str):
        return "OTHER"
    s = name.lower()

    # NVIDIA patterns
    nvidia_keys = ["rtx", "gtx", "geforce", "quadro", "tesla", "titan"]
    if any(k in s for k in nvidia_keys):
        return "NVIDIA"

    # AMD patterns
    amd_keys = ["radeon", "rx ", "rx-", "r9 ", "r7 ", "r5 ", "vega"]
    if any(k in s for k in amd_keys):
        return "AMD"

    # INTEL patterns
    intel_keys = ["intel", "arc a", "uhd graphics", "iris xe"]
    if any(k in s for k in intel_keys):
        return "INTEL"

    return "OTHER"


@st.cache_data
def load_historical_prices() -> pd.DataFrame:
    path = DATA_DIR / "gpu_price_history.csv"
    df = pd.read_csv(path)

    df = df.rename(
        columns={
            "Date": "date",
            "Name": "name",
            "Retail Price": "price_new",
            "Used Price": "price_used",
        }
    )

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["vendor"] = df["name"].apply(infer_vendor_from_name)

    for col in ["price_new", "price_used"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(0, np.nan)

    df = df[df["vendor"].isin(["NVIDIA", "AMD", "INTEL"])]
    df = df.dropna(subset=["date"])

    return df


@st.cache_data
def load_latest_specs() -> pd.DataFrame:
    path = DATA_DIR / "gpu_specs_prices.csv"
    df = pd.read_csv(path, encoding="latin-1")

    if "chipset" in df.columns:
        df["vendor"] = df["chipset"].apply(infer_vendor_from_name)
    elif "model" in df.columns:
        df["vendor"] = df["model"].apply(infer_vendor_from_name)
    else:
        df["vendor"] = df["name"].apply(infer_vendor_from_name)

    if "price" in df.columns:
        df["price_raw"] = df["price"]
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df["price"] = pd.to_numeric(df["price"], errors="coerce")

    return df


@st.cache_data
def load_shipments() -> pd.DataFrame:
    path = DATA_DIR / "gpu_shipments_market_share.csv"
    df = pd.read_csv(path)

    df["vendor"] = df["vendor"].str.upper()
    df["units_mln"] = pd.to_numeric(df["units_mln"], errors="coerce")
    df["market_share"] = pd.to_numeric(df["market_share"], errors="coerce")

    df["period"] = pd.PeriodIndex(
        year=df["year"].astype(int),
        quarter=df["quarter"].astype(int),
        freq="Q"
    ).to_timestamp(how="start")

    return df


@st.cache_data
def load_price_index_table() -> pd.DataFrame:
    path = DATA_DIR / "GPU_Price_Index.csv"
    df = pd.read_csv(path)

    for col in ["Best US Price", "Lowest-Ever US Price"]:
        df[col + " (num)"] = (
            df[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        df[col + " (num)"] = pd.to_numeric(df[col + " (num)"], errors="coerce")

    return df
def build_aggregated_price_index(
    df_hist: pd.DataFrame,
    vendors: list[str],
    price_column: str,
    freq: str,
) -> pd.DataFrame:
    df = df_hist.copy()
    df = df[df["vendor"].isin(vendors)]

    if price_column not in df.columns:
        raise ValueError(f"Column {price_column} not found in dataset")

    df = df.dropna(subset=[price_column])
    df["period"] = df["date"].dt.to_period(freq).dt.to_timestamp()

    grouped = (
        df.groupby(["period", "vendor"])[price_column]
        .mean()
        .reset_index()
        .rename(columns={price_column: "avg_price"})
    )

    return grouped


def build_forecast(series: pd.Series, horizon: int):
    s = series.dropna().sort_index()
    if len(s) < 6:
        raise ValueError("Not enough data points for forecasting (min 6).")

    model = ExponentialSmoothing(
        s,
        trend="add",
        seasonal=None,
        initialization_method="estimated",
    ).fit(optimized=True)

    forecast = model.forecast(horizon)
    return forecast, model


def apply_scenario(forecast: pd.Series, pct: float) -> pd.Series:
    return forecast * (1 + pct / 100.0)


# ------------------------------------------------------
#  LOAD DATA
# ------------------------------------------------------

with st.spinner("Loading datasets..."):
    hist_df = load_historical_prices()
    latest_df = load_latest_specs()
    price_index_df = load_price_index_table()
    shipments_df = None
    try:
        shipments_df = load_shipments()
    except Exception:
        shipments_df = None

# ------------------------------------------------------
#  SIDEBAR
# ------------------------------------------------------

st.sidebar.title("⚙️ Settings")

vendor_options = ["NVIDIA", "AMD", "INTEL"]
selected_vendors = st.sidebar.multiselect(
    "Vendors:",
    vendor_options,
    default=["NVIDIA", "AMD"],
)

price_type = st.sidebar.radio(
    "Historical price type:",
    ["price_new", "price_used"],
    format_func=lambda x: "Retail price" if x == "price_new" else "Used price",
)

time_agg = st.sidebar.radio(
    "Time aggregation:",
    ["M", "Q"],
    format_func=lambda x: "Monthly" if x == "M" else "Quarterly",
)

forecast_horizon = st.sidebar.slider(
    "Forecast horizon (steps):",
    min_value=2,
    max_value=12,
    value=6,
    step=1,
)

scenario_pct = st.sidebar.slider(
    "Scenario adjustment (%):",
    -30,
    30,
    0,
    1,
)

# ------------------------------------------------------
#  TABS
# ------------------------------------------------------

tab_overview, tab_forecast, tab_shipments, tab_index, tab_data = st.tabs(
    [
        "📊 Market overview",
        "📈 Price dynamics & forecast",
        "🌍 Shipments & market share",
        "🏷️ US price index",
        "🗂 Data",
    ]
)

# ------------------------------------------------------
#  TAB 1 — OVERVIEW
# ------------------------------------------------------

with tab_overview:
    st.header("📊 Current GPU market overview")

    latest_filtered = latest_df[latest_df["vendor"].isin(selected_vendors)]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Models in dataset", len(latest_filtered))
    with c2:
        if "price" in latest_filtered.columns and latest_filtered["price"].notna().any():
            avg_price = latest_filtered["price"].mean()
            st.metric("Average price", f"{avg_price:.0f} $")
        else:
            st.metric("Average price", "N/A")
    with c3:
        if "memory" in latest_filtered.columns:
            mem_num = latest_filtered["memory"].astype(str).str.extract(r"(\d+)")
            mem_num = pd.to_numeric(mem_num[0], errors="coerce")
            if mem_num.notna().any():
                st.metric("Median VRAM", f"{mem_num.median():.1f} GB")
            else:
                st.metric("Median VRAM", "N/A")
        else:
            st.metric("Median VRAM", "N/A")

    st.subheader("Price distribution by vendor")
    if "price" in latest_filtered.columns and latest_filtered["price"].notna().any():
        fig_hist = px.histogram(
            latest_filtered,
            x="price",
            color="vendor",
            nbins=50,
            marginal="box",
            template="plotly_dark",
            title="GPU price distribution",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No valid price data found in gpu_specs_prices.csv")

    st.subheader("Models table")
    st.dataframe(
        latest_filtered.sort_values(by=["vendor", "price"], ascending=True),
        use_container_width=True,
        height=450,
    )


# ------------------------------------------------------
#  TAB 2 — FORECAST
# ------------------------------------------------------

with tab_forecast:
    st.header("📈 Historical price dynamics & forecast")

    try:
        idx_df = build_aggregated_price_index(
            hist_df,
            selected_vendors,
            price_type,
            time_agg,
        )
    except Exception as e:
        st.error(f"Failed to build price index: {e}")
        idx_df = pd.DataFrame()

    if idx_df.empty:
        st.warning("No valid price data for selected vendors.")
    else:
        fig_idx = px.line(
            idx_df,
            x="period",
            y="avg_price",
            color="vendor",
            markers=True,
            template="plotly_dark",
            title="Historical price index",
        )
        st.plotly_chart(fig_idx, use_container_width=True)

        total_series = (
            idx_df.groupby("period")["avg_price"]
            .mean()
            .rename("index_value")
        )

        st.subheader("Forecast of price index")

        try:
            base_forecast, model = build_forecast(total_series, forecast_horizon)
            adj_forecast = apply_scenario(base_forecast, scenario_pct)

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=total_series.index,
                y=total_series.values,
                mode="lines+markers",
                name="Historical index",
            ))
            fig_f.add_trace(go.Scatter(
                x=adj_forecast.index,
                y=adj_forecast.values,
                mode="lines+markers",
                name="Forecast (adjusted)",
            ))
            fig_f.update_layout(
                template="plotly_dark",
                xaxis_title="Period",
                yaxis_title="Index value",
            )
            st.plotly_chart(fig_f, use_container_width=True)

        except Exception as e:
            st.error(f"Forecast error: {e}")
            
# ------------------------------------------------------
#  TAB 3 — SHIPMENTS
# ------------------------------------------------------

with tab_shipments:
    st.header("🌍 Discrete GPU shipments & market share")

    if shipments_df is None:
        st.warning(
            "gpu_shipments_market_share.csv was not found in /data/. "
            "Please place the file into the data/ folder."
        )
    else:
        ship_filtered = shipments_df[shipments_df["vendor"].isin(selected_vendors)]

        if ship_filtered.empty:
            st.warning("No shipment data available for selected vendors.")
        else:
            st.subheader("Quarterly shipments (in million units)")
            fig_units = px.bar(
                ship_filtered,
                x="period",
                y="units_mln",
                color="vendor",
                barmode="group",
                template="plotly_dark",
                labels={"period": "Quarter", "units_mln": "Units (mln)"},
            )
            st.plotly_chart(fig_units, use_container_width=True)

            st.subheader("Market share timeline")
            fig_share = px.area(
                ship_filtered,
                x="period",
                y="market_share",
                color="vendor",
                template="plotly_dark",
                labels={"period": "Quarter", "market_share": "Share (%)"},
            )
            st.plotly_chart(fig_share, use_container_width=True)

            st.subheader("Raw shipment data")
            st.dataframe(
                ship_filtered.sort_values(by=["period", "vendor"]),
                use_container_width=True,
                height=400,
            )


# ------------------------------------------------------
#  TAB 4 — GPU PRICE INDEX TABLE
# ------------------------------------------------------

with tab_index:
    st.header("🏷️ US GPU price index (best & lowest-ever prices)")

    if price_index_df.empty:
        st.warning("Could not load GPU_Price_Index.csv")
    else:
        st.subheader("Price index table")
        st.dataframe(
            price_index_df,
            use_container_width=True,
            height=450,
        )

        num_best = "Best US Price (num)"
        num_low = "Lowest-Ever US Price (num)"
        name_col = "GPU Model"

        if num_best in price_index_df.columns and num_low in price_index_df.columns:
            st.subheader("Best vs lowest-ever GPU price (top 20 models)")

            top_df = (
                price_index_df[[name_col, num_best, num_low]]
                .dropna()
                .sort_values(by=num_best, ascending=False)
                .head(20)
            )

            fig_idx_price = go.Figure()
            fig_idx_price.add_trace(go.Bar(
                x=top_df[name_col],
                y=top_df[num_best],
                name="Best US price",
            ))
            fig_idx_price.add_trace(go.Bar(
                x=top_df[name_col],
                y=top_df[num_low],
                name="Lowest-ever US price",
            ))

            fig_idx_price.update_layout(
                template="plotly_dark",
                barmode="group",
                xaxis_tickangle=-45,
                title="Best vs Lowest-ever Prices",
            )
            st.plotly_chart(fig_idx_price, use_container_width=True)

        else:
            st.info(
                "Numeric versions of price columns were not created. "
                "Check column names in GPU_Price_Index.csv"
            )


# ------------------------------------------------------
#  TAB 5 — RAW DATA TABLES
# ------------------------------------------------------

with tab_data:
    st.header("🗂 Source datasets")

    dataset_choice = st.selectbox(
        "Select dataset to view:",
        [
            "Historical prices (gpu_price_history.csv)",
            "Current specs & prices (gpu_specs_prices.csv)",
            "Shipments & market share (gpu_shipments_market_share.csv)",
            "US price index (GPU_Price_Index.csv)",
        ],
    )

    if dataset_choice.startswith("Historical"):
        st.subheader("Historical GPU prices")
        st.dataframe(hist_df, use_container_width=True, height=450)

    elif dataset_choice.startswith("Current"):
        st.subheader("Current GPU specs & prices")
        st.dataframe(latest_df, use_container_width=True, height=450)

    elif dataset_choice.startswith("Shipments"):
        if shipments_df is None:
            st.warning("gpu_shipments_market_share.csv not loaded.")
        else:
            st.subheader("GPU shipments data")
            st.dataframe(shipments_df, use_container_width=True, height=450)

    else:
        st.subheader("US price index table")
        st.dataframe(price_index_df, use_container_width=True, height=450)

