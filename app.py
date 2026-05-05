import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pathlib import Path

# ======================================================
#   PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="GPU Market Analytics",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================
#   CUSTOM CSS (READABILITY + DARK THEME)
# ======================================================

CUSTOM_CSS = """
<style>
    /* Общий тёмный фон */
    html, body, .stApp {
        background-color: #0f1116 !important;
        color: #ffffff !important;
    }

    /* Контейнер основного приложения */
    [data-testid="stAppViewContainer"] {
        background-color: #0f1116 !important;
    }

    /* Убираем белый хедер Streamlit */
    [data-testid="stHeader"] {
        background-color: transparent !important;
        box-shadow: none !important;
    }

    /* Основной контейнер контента */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    .metric-card {
        background-color: #1b1e27;
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid #2b2f3b;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a3b1;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #ffffff;
    }

    .stTabs [role="tab"] {
        background-color: #1b1e27;
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #3b3f50;
        color: #ffffff;
    }

    /* Сайдбар */
    section[data-testid="stSidebar"] {
        background-color: #14161d !important;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
        font-size: 0.95rem !important;
    }
    section[data-testid="stSidebar"] label {
        font-weight: 600 !important;
    }


    [data-testid="collapsedControl"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
        opacity: 1 !important;
    }

    /* Поля selectbox/multiselect — общие стили полей */
    div[data-testid="stSelectbox"] > div,
    div[data-testid="stMultiSelect"] > div {
        background-color: #1b1e27 !important;
        border: 1px solid #2b2f3b !important;
        border-radius: 8px !important;
    }

    /* Текст внутри селектов в сайдбаре — белый (не трогаем сайдбар) */
    section[data-testid="stSidebar"] div[data-testid="stSelectbox"] > div * ,
    section[data-testid="stSidebar"] div[data-testid="stMultiSelect"] > div * {
        color: #ffffff !important;
    }

    [data-testid="stAppViewContainer"] div[data-testid="stSelectbox"] > div * ,
    [data-testid="stAppViewContainer"] div[data-testid="stMultiSelect"] > div * {
        color: #000000 !important;
    }

    /* Стрелочка у селектов (всегда видима, белая) */
    div[data-testid="stSelectbox"] svg,
    div[data-testid="stMultiSelect"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
        opacity: 1 !important;
    }

    /* Лёгкий hover для селектов */
    div[data-testid="stSelectbox"] > div:hover,
    div[data-testid="stMultiSelect"] > div:hover {
        border-color: #50576b !important;
    }

    div[role="listbox"] [role="option"],
    div[role="listbox"] [role="option"] *,
    ul[role="listbox"] li,
    ul[role="listbox"] li * {
        color: #000000 !important;            
        background-color: #ffffff !important; 
    }
    div[role="listbox"] [role="option"]:hover,
    ul[role="listbox"] li:hover {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
    }
    [data-baseweb="select"] li, [data-baseweb="select"] li * {
        color: #000000 !important;
        background-color: #ffffff !important;
    }

</style>
"""



st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DATA_DIR = Path("data")

# ======================================================
#   UNIVERSAL CSV LOADER (ФІКС КОДУВАННЯ)
# ======================================================

def read_csv_safely(path: Path) -> pd.DataFrame:
    """
    Універсальне читання CSV з різними кодуваннями.
    Спочатку пробуємо UTF-8, далі cp1251, далі latin1.
    """
    encodings = ["utf-8", "cp1251", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
            continue
    # fallback з ігноруванням помилок
    return pd.read_csv(path, encoding="latin1", errors="ignore")


# ======================================================
#   HELPER FUNCTIONS
# ======================================================

def infer_vendor_from_name(name: str) -> str:
    """
    Автоматичне визначення вендора за назвою моделі.
    Повертає NVIDIA / AMD / INTEL / OTHER.
    """
    if not isinstance(name, str):
        return "OTHER"
    s = name.lower()

    nvidia_keys = ["rtx", "gtx", "geforce", "titan", "quadro"]
    amd_keys = ["radeon", "rx", "vega", "r9", "r7"]
    intel_keys = ["arc", "intel", "iris", "uhd"]

    if any(k in s for k in nvidia_keys):
        return "NVIDIA"
    if any(k in s for k in amd_keys):
        return "AMD"
    if any(k in s for k in intel_keys):
        return "INTEL"
    return "OTHER"


@st.cache_data(show_spinner=False)
def load_historical_prices() -> pd.DataFrame:
    """
    Історичні ціни відеокарт:
    gpu_price_history.csv: Date, Name, Retail Price, Used Price
    """
    file_path = DATA_DIR / "gpu_price_history.csv"
    if not file_path.exists():
        st.warning("File gpu_price_history.csv not found in /data/.")
        return pd.DataFrame()

    df = read_csv_safely(file_path)

    # Перейменування колонок до єдиного стандарту
    rename_map = {
        "Date": "date",
        "Name": "name",
        "Retail Price": "price_new",
        "Used Price": "price_used",
    }
    df = df.rename(columns=rename_map)

    # Безпечний парсинг дати: спочатку жорсткий формат, потім fallback
    if "date" not in df.columns:
        st.error("Column 'Date' not found in gpu_price_history.csv")
        return pd.DataFrame()

    date_raw = df["date"].astype(str).str.strip()
    # Основний варіант: dd.mm.yyyy
    parsed = pd.to_datetime(date_raw, format="%d.%m.%Y", errors="coerce")
    # Fallback на випадок інших форматів (dayfirst=True)
    mask = parsed.isna() & date_raw.notna()
    if mask.any():
        parsed_fallback = pd.to_datetime(
            date_raw[mask],
            errors="coerce",
            dayfirst=True,
        )
        parsed.loc[mask] = parsed_fallback

    df["date"] = parsed
    df = df.dropna(subset=["date"])

    # Вендор
    if "name" not in df.columns:
        st.error("Column 'Name' not found in gpu_price_history.csv")
        return pd.DataFrame()
    df["vendor"] = df["name"].apply(infer_vendor_from_name)

    # Ціни
    for col in ["price_new", "price_used"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace(0, np.nan)

    df = df.sort_values("date")
    return df


@st.cache_data(show_spinner=False)
def load_latest_specs() -> pd.DataFrame:
    """
    Актуальні моделі та ціни: gpu_specs_prices.csv.
    """
    file_path = DATA_DIR / "gpu_specs_prices.csv"
    if not file_path.exists():
        st.warning("File gpu_specs_prices.csv not found in /data/.")
        return pd.DataFrame()

    df = read_csv_safely(file_path)

    # Назва моделі
    name_candidates = ["name", "model", "GPU Name", "Product"]
    name_col = next((c for c in name_candidates if c in df.columns), None)
    if name_col is None:
        st.error("Could not find GPU model name column in gpu_specs_prices.csv")
        return pd.DataFrame()
    df = df.rename(columns={name_col: "name"})

    # Вендор
    if "vendor" not in df.columns:
        chipset_candidates = ["chipset", "Chipset", "GPU Chip"]
        chipset_col = next((c for c in chipset_candidates if c in df.columns), None)
        if chipset_col:
            df["vendor"] = df[chipset_col].apply(infer_vendor_from_name)
        else:
            df["vendor"] = df["name"].apply(infer_vendor_from_name)
    else:
        df["vendor"] = df["vendor"].fillna("").apply(infer_vendor_from_name)

    # Ціна
    price_candidates = ["price", "Price", "USD Price", "MSRP"]
    price_col = next((c for c in price_candidates if c in df.columns), None)
    if price_col is None:
        st.error("Could not find price column in gpu_specs_prices.csv")
        return pd.DataFrame()

    df[price_col] = (
        df[price_col]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df["price_usd"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["price_usd"])
    df = df[df["price_usd"] > 0]

    return df


@st.cache_data(show_spinner=False)
def load_shipments() -> pd.DataFrame | None:
    """
    Дані по поставках та ринкових частках: gpu_shipments_market_share.csv
    """
    file_path = DATA_DIR / "gpu_shipments_market_share.csv"
    if not file_path.exists():
        return None

    df = read_csv_safely(file_path)

    required = {"year", "quarter", "vendor", "units_mln", "market_share"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Missing columns in gpu_shipments_market_share.csv: {missing}")
        return None

    df["period"] = pd.PeriodIndex(
        year=df["year"], quarter=df["quarter"], freq="Q"
    ).to_timestamp()

    df["units_mln"] = pd.to_numeric(df["units_mln"], errors="coerce")
    df["market_share"] = pd.to_numeric(df["market_share"], errors="coerce")
    df = df.dropna(subset=["units_mln", "market_share"])
    df = df.sort_values("period")

    return df


@st.cache_data(show_spinner=False)
def load_price_index() -> pd.DataFrame:
    """
    Додатковий датасет з індексами цін (GPU_Price_Index.csv).
    """
    file_path = DATA_DIR / "GPU_Price_Index.csv"
    if not file_path.exists():
        return pd.DataFrame()

    df = read_csv_safely(file_path)

    # Підготовка числових версій цін
    for col in ["Best US Price", "Lowest-Ever US Price"]:
        if col in df.columns:
            num_col = col + " (num)"
            df[num_col] = (
                df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    return df


def build_aggregated_price_index(
    df_hist: pd.DataFrame,
    vendors: list[str],
    price_column: str,
    freq: str,
) -> pd.DataFrame:
    """
    Агрегований індекс цін за вендорами з обраною частотою (M/Q).
    """
    if df_hist.empty:
        return pd.DataFrame()

    df = df_hist[df_hist["vendor"].isin(vendors)].copy()
    if price_column not in df.columns:
        raise ValueError(f"Column {price_column} not found in historical dataset")

    df = df.dropna(subset=[price_column])
    if df.empty:
        return pd.DataFrame()

    df["period"] = df["date"].dt.to_period(freq).dt.to_timestamp()

    grouped = (
        df.groupby(["period", "vendor"])[price_column]
        .mean()
        .reset_index()
        .rename(columns={price_column: "avg_price"})
    )
    return grouped


def prepare_series_for_forecast(
    df_hist: pd.DataFrame,
    vendors: list[str],
    price_column: str,
    freq: str,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Готує історичний ряд для побудови прогнозу:
    - агрегує індекс,
    - усереднює по вендорах,
    - відсікає дані після 01.01.2026 (щоб прогноз починався з 2026 року).
    """
    idx_df = build_aggregated_price_index(df_hist, vendors, price_column, freq)
    if idx_df.empty:
        return pd.Series(dtype="float64"), idx_df

    series = (
        idx_df.groupby("period")["avg_price"]
        .mean()
        .rename("index_value")
        .sort_index()
    )

    cutoff = pd.Timestamp("2026-01-01")
    hist = series[series.index < cutoff]
    if hist.empty:
        hist = series

    return hist, idx_df


def apply_scenario(forecast: pd.Series, pct: float) -> pd.Series:
    """
    Сценарне коригування прогнозу на +/- %.
    """
    return forecast * (1 + pct / 100.0)


def build_forecast_with_model(
    series: pd.Series,
    horizon: int,
    model_name: str,
    freq: str,
):
    """
    Побудова прогнозу однією з моделей:
    - Holt-Winters (trend + seasonality)
    - Holt-Winters (damped trend)
    - ARIMA(1,1,1)
    - SARIMA(1,1,1)x(1,1,1,s)
    """
    s = series.dropna().sort_index()
    # безопасный fallback: для Monthly требуем больше точек, для Quarterly — чуть меньше
    min_points = 12 if freq == "M" else 8
    if len(s) < min_points:
        # предупреждаем пользователя и возвращаем простой линейный прогноз
        st.warning(f"Not enough data points ({len(s)}) for chosen aggregation '{freq}'. Using linear fallback.")
        x = np.arange(len(s))
        if len(x) == 0:
            raise ValueError(f"Not enough data points for forecasting (need at least {min_points}).")
        coef = np.polyfit(x, s.values, 1)
        pred_x = np.arange(len(s), len(s) + horizon)
        vals = np.polyval(coef, pred_x)
        if isinstance(s.index, pd.DatetimeIndex):
            freq_map = {"M": "MS", "Q": "QS"}
            tgt_freq = freq_map.get(freq)
            forecast = pd.Series(vals, index=pd.date_range(start=s.index[-1], periods=horizon + 1, freq=tgt_freq)[1:])
        else:
            forecast = pd.Series(vals)
        return forecast, None

    seasonal_periods = 12 if freq == "M" else 4

    if model_name == "Trend + seasonality (Holt-Winters)":
        model = ExponentialSmoothing(
            s,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)
        forecast = model.forecast(horizon)

    elif model_name == "Damped trend (Holt-Winters)":
        model = ExponentialSmoothing(
            s,
            trend="add",
            damped_trend=True,
            seasonal=None,
            initialization_method="estimated",
        ).fit(optimized=True)
        forecast = model.forecast(horizon)
# ...existing code...
    elif model_name == "ARIMA (1,1,1)":
        # Для ARIMA с d=1 нельзя ставить 'c' (константу) — используем линейный тренд 't' (drift).
        # Сначала приведём индекс к нужной частоте (MS/QS), если это DatetimeIndex, и заполним пропуски.
        if isinstance(s.index, pd.DatetimeIndex):
            freq_map = {"M": "MS", "Q": "QS"}
            tgt_freq = freq_map.get(freq, None)
            if tgt_freq is not None:
                if s.index.freq is None or s.index.freqstr != tgt_freq:
                    s = s.asfreq(tgt_freq)
                    s = s.interpolate(method="time")

        # Фитим ARIMA с линейным трендом (slope) — он допустим при d=1
        model = ARIMA(s, order=(1, 1, 1), trend="t").fit()
        fc = model.get_forecast(steps=horizon).predicted_mean

        # Назначаем корректный datetime-индекс прогнозу, чтобы фильтр >= 2026 работал
        if isinstance(s.index, pd.DatetimeIndex):
            last = s.index[-1]
            if freq == "M":
                idx = pd.date_range(start=last, periods=horizon + 1, freq="MS")[1:]
            else:  # "Q"
                idx = pd.date_range(start=last, periods=horizon + 1, freq="QS")[1:]
            fc.index = idx

        forecast = fc
# ...existing code...

    elif model_name == "SARIMA (1,1,1)x(1,1,1,s)":
        model = SARIMAX(
            s,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal_periods),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        forecast = model.forecast(steps=horizon)

    else:
        raise ValueError("Unknown model type.")

    return forecast, model


# ======================================================
#   LOAD ALL DATA
# ======================================================

with st.spinner("Loading datasets..."):
    hist_df = load_historical_prices()
    latest_df = load_latest_specs()
    shipments_df = load_shipments()
    price_index_df = load_price_index()

# ======================================================
#   SIDEBAR CONTROLS
# ======================================================

st.sidebar.title("⚙ Settings")

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
    index=0,
    format_func=lambda x: "Monthly" if x == "M" else "Quarterly",
)

forecast_horizon_months = st.sidebar.slider(
    "Forecast horizon (months):",
    min_value=24,
    max_value=36,
    value=30,
    step=1,
    help="Forecast length for the separate forecasting module (from 2026).",
)

scenario_pct = st.sidebar.slider(
    "Scenario adjustment (%):",
    min_value=-30,
    max_value=30,
    value=0,
    step=1,
    help="Apply positive or negative scenario to the forecast.",
)

# ======================================================
#   TABS
# ======================================================

(
    tab_overview,
    tab_price_dyn,
    tab_shipments,
    tab_us_index,
    tab_forecast_models,
    tab_data,
) = st.tabs(
    [
        "📊 Market overview",
        "📈 Price dynamics",
        "🌍 Shipments & market share",
        "🏷️ US price index",
        "🔮 Forecast models",
        "🗂 Data",
    ]
)

# ======================================================
#   TAB 1 — MARKET OVERVIEW
# ======================================================

with tab_overview:
    st.header("📊 Current GPU market overview")

    if latest_df.empty:
        st.warning("Specs & prices dataset is not available.")
    else:
        latest_filtered = latest_df[latest_df["vendor"].isin(selected_vendors)]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("##### Vendors in dataset")
            if latest_filtered.empty:
                st.write("No data for selected vendors.")
            else:
                st.write(", ".join(sorted(latest_filtered["vendor"].unique())))

        with c2:
            st.markdown("##### Number of models")
            st.write(int(latest_filtered["name"].nunique()) if not latest_filtered.empty else 0)

        with c3:
            st.markdown("##### Price range (USD)")
            if not latest_filtered.empty:
                st.write(
                    f"{latest_filtered['price_usd'].min():.0f} – "
                    f"{latest_filtered['price_usd'].max():.0f}"
                )
            else:
                st.write("—")

        st.markdown("---")

        if latest_filtered.empty:
            st.info("Change vendor selection in the sidebar to see models.")
        else:
            st.subheader("Price distribution of current GPUs")
            fig_hist = px.histogram(
                latest_filtered,
                x="price_usd",
                color="vendor",
                nbins=40,
                template="plotly_dark",
                marginal="box",
                title="Distribution of current GPU prices by vendor",
            )
            fig_hist.update_layout(
                xaxis_title="Price, USD",
                yaxis_title="Number of models",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Sample of current GPUs (top 50 by price)")
            top_models = (
                latest_filtered.sort_values(by="price_usd", ascending=False)
                .head(50)
                .reset_index(drop=True)
            )
            st.dataframe(top_models, use_container_width=True, height=450)

# ======================================================
#   TAB 2 — HISTORICAL PRICE DYNAMICS (WITHOUT FORECAST)
# ======================================================

with tab_price_dyn:
    st.header("📈 Historical price dynamics")

    if hist_df.empty:
        st.warning("Historical dataset is not available.")
    else:
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
            st.subheader("Price index by vendor")
            fig_idx = px.line(
                idx_df,
                x="period",
                y="avg_price",
                color="vendor",
                markers=True,
                template="plotly_dark",
                title="Historical price index for selected vendors",
            )
            fig_idx.update_layout(
                xaxis_title="Period",
                yaxis_title="Average price",
            )
            st.plotly_chart(fig_idx, use_container_width=True)

            total_series = (
                idx_df.groupby("period")["avg_price"]
                .mean()
                .rename("index_value")
                .reset_index()
            )

            st.subheader("Aggregated market price index")
            fig_total = px.line(
                total_series,
                x="period",
                y="index_value",
                markers=True,
                template="plotly_dark",
                title="Average price index across selected vendors",
            )
            fig_total.update_layout(
                xaxis_title="Period",
                yaxis_title="Average price index",
            )
            st.plotly_chart(fig_total, use_container_width=True)

            st.caption(
                "This tab shows historical dynamics only. "
                "Forecasts are available on the 'Forecast models' tab."
            )

# ======================================================
#   TAB 3 — SHIPMENTS & MARKET SHARE
# ======================================================

with tab_shipments:
    st.header("🌍 Discrete GPU shipments & market share")

    if shipments_df is None:
        st.warning("Shipments dataset (gpu_shipments_market_share.csv) is not available.")
    else:
        ship_filtered = shipments_df[shipments_df["vendor"].isin(selected_vendors)]

        if ship_filtered.empty:
            st.warning("No shipment data for selected vendors.")
        else:
            st.subheader("Quarterly shipments (in million units)")
            fig_units = px.bar(
                ship_filtered,
                x="period",
                y="units_mln",
                color="vendor",
                barmode="group",
                template="plotly_dark",
                title="Discrete GPU shipments by vendor",
            )
            fig_units.update_layout(
                xaxis_title="Quarter",
                yaxis_title="Units, mln",
            )
            st.plotly_chart(fig_units, use_container_width=True)

            st.subheader("Market share over time")
            fig_share = px.area(
                ship_filtered,
                x="period",
                y="market_share",
                color="vendor",
                groupnorm="fraction",
                template="plotly_dark",
                title="Vendor market share over time",
            )
            fig_share.update_layout(
                xaxis_title="Quarter",
                yaxis_title="Share",
            )
            st.plotly_chart(fig_share, use_container_width=True)

# ======================================================
#   TAB 4 — US PRICE INDEX DATASET
# ======================================================

with tab_us_index:
    st.header("🏷️ US GPU price index (best & lowest-ever prices)")

    if price_index_df.empty:
        st.warning("GPU_Price_Index.csv not found in /data/.")
    else:
        st.subheader("Price index table")
        st.dataframe(price_index_df, use_container_width=True, height=450)

        num_best = "Best US Price (num)"
        num_low = "Lowest-Ever US Price (num)"
        name_col = "GPU Model" if "GPU Model" in price_index_df.columns else None

        if name_col and num_best in price_index_df.columns and num_low in price_index_df.columns:
            st.subheader("The most popular top 20 models")
            top_df = (
                price_index_df[[name_col, num_best, num_low]]
                .dropna()
                .sort_values(by=num_best, ascending=False)
                .head(20)
            )

            fig_idx_price = go.Figure()
            fig_idx_price.add_trace(
                go.Bar(
                    x=top_df[name_col],
                    y=top_df[num_best],
                    name="Best price",
                )
            )
            fig_idx_price.add_trace(
                go.Bar(
                    x=top_df[name_col],
                    y=top_df[num_low],
                    name="Lowest-ever price",
                )
            )
            fig_idx_price.update_layout(
                barmode="group",
                template="plotly_dark",
                xaxis_title="GPU model",
                yaxis_title="Price, USD",
            )
            st.plotly_chart(fig_idx_price, use_container_width=True)

# ======================================================
#   TAB 5 — FORECAST MODELS (SEPARATE FORECAST MODULE)
# ======================================================

with tab_forecast_models:
    st.header("🔮 Forecast of GPU price index starting from 2026")

    st.markdown(
        "Use the controls in the sidebar to choose **vendors**, "
        "**price type** and **time aggregation**. "
        "Then select a **forecast model** and click **Run forecast**."
    )

    model_name = st.selectbox(
        "Forecast model:",
        [
            "Trend + seasonality (Holt-Winters)",
            "Damped trend (Holt-Winters)",
            "ARIMA (1,1,1)",
            "SARIMA (1,1,1)x(1,1,1,s)",
        ],
    )

    if st.button("Run forecast"):
        if hist_df.empty:
            st.error("Historical dataset is not available.")
        else:
            try:
                series, idx_df = prepare_series_for_forecast(
                    hist_df,
                    selected_vendors,
                    price_type,
                    time_agg,
                )
            except Exception as e:
                st.error(f"Failed to prepare series for forecast: {e}")
                series = pd.Series(dtype="float64")
                idx_df = pd.DataFrame()

            if idx_df.empty or series.empty:
                st.warning("Not enough data to build a forecast for the selected settings.")
            else:
                # кроки прогнозу (переведення місяців у періоди)
                if time_agg == "M":
                    horizon_steps = forecast_horizon_months
                else:  # "Q"
                    horizon_steps = max(2, forecast_horizon_months // 3)

                try:
                    base_forecast, model = build_forecast_with_model(
                        series,
                        horizon_steps,
                        model_name,
                        time_agg,
                    )
                    # залишаємо лише значення починаючи з 2026 року
                    base_forecast = base_forecast[base_forecast.index >= pd.Timestamp("2026-01-01")]
                    if base_forecast.empty:
                        st.warning(
                            "The forecast horizon is too short to reach 2026. "
                            "Increase the horizon in months in the sidebar."
                        )
                    else:
                        adj_forecast = apply_scenario(base_forecast, scenario_pct)

                        fig_f = go.Figure()
                        fig_f.add_trace(
                            go.Scatter(
                                x=series.index,
                                y=series.values,
                                mode="lines+markers",
                                name="Historical index",
                            )
                        )
                        fig_f.add_trace(
                            go.Scatter(
                                x=adj_forecast.index,
                                y=adj_forecast.values,
                                mode="lines+markers",
                                name="Forecast (adjusted)",
                            )
                        )
                        fig_f.update_layout(
                            template="plotly_dark",
                            xaxis_title="Period",
                            yaxis_title="Index value",
                        )
                        st.plotly_chart(fig_f, use_container_width=True)

                        st.markdown(
                            f"**Model:** {model_name}  \n"
                            f"**Forecast from:** {adj_forecast.index.min().date()}  \n"
                            f"**Forecast to:** {adj_forecast.index.max().date()}  \n"
                            f"**Scenario adjustment:** {scenario_pct}%"
                        )

                except Exception as e:
                    st.error(f"Forecast error: {e}")
    else:
        st.info("Select a model and click **Run forecast** to build a forecast starting from 2026.")

# ======================================================
#   TAB 6 — RAW DATA
# ======================================================

with tab_data:
    st.header("🗂 Source datasets")

    st.subheader("Historical GPU price history")
    if hist_df.empty:
        st.info("Historical dataset is not available.")
    else:
        st.dataframe(hist_df.head(1000), use_container_width=True, height=350)

    st.subheader("Current GPU specs & prices")
    if latest_df.empty:
        st.info("Specs & prices dataset is not available.")
    else:
        st.dataframe(latest_df.head(1000), use_container_width=True, height=350)

    st.subheader("Shipments & market share")
    if shipments_df is None:
        st.info("Shipments dataset is not available.")
    else:
        st.dataframe(shipments_df.head(1000), use_container_width=True, height=350)

    st.subheader("US price index dataset")
    if price_index_df.empty:
        st.info("US price index dataset is not available.")
    else:
        st.dataframe(price_index_df.head(1000), use_container_width=True, height=350)

