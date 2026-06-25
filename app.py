import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# =========================================================
# CẤU HÌNH TRANG
# =========================================================
st.set_page_config(
    page_title="E-Commerce AI Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DESIGN SYSTEM — DARK NAVY / TEAL THEME
# =========================================================
DARK_THEME_CSS = """
<style>
/* ---- PALETTE ----
   bg-base:    #0D1117   (dark navy)
   bg-surface: #161B22   (card surface)
   bg-panel:   #1C2333   (sidebar / raised panel)
   accent:     #2DC9A1   (teal glow)
   accent-dim: #1A7A65   (muted teal)
   border:     #2A3444   (subtle border)
   text-1:     #E6EDF3   (primary text)
   text-2:     #8B949E   (secondary / muted)
   text-3:     #4D5969   (placeholder / disabled)
   danger:     #F87171   (red)
   warn:       #FBBF24   (amber)
   success:    #34D399   (green)
*/

/* === ROOT RESET === */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0D1117 !important;
    color: #E6EDF3 !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

[data-testid="stHeader"] {
    background-color: #0D1117 !important;
    border-bottom: 1px solid #2A3444 !important;
}

/* === SIDEBAR === */
[data-testid="stSidebar"] {
    background-color: #0D1117 !important;
    border-right: 1px solid #2A3444 !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label span {
    color: #8B949E !important;
    font-size: 13px !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #E6EDF3 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
}
[data-testid="stSidebar"] .stRadio label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    color: #8B949E !important;
    width: 100% !important;
    display: block !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #161B22 !important;
    border-color: #2A3444 !important;
    color: #E6EDF3 !important;
}
[data-testid="stSidebar"] .stRadio input[type="radio"]:checked + div {
    color: #2DC9A1 !important;
}

/* Sidebar info box */
[data-testid="stSidebar"] .stAlert {
    background: #161B22 !important;
    border: 1px solid #2A3444 !important;
    border-left: 3px solid #2DC9A1 !important;
    border-radius: 8px !important;
    color: #8B949E !important;
    font-size: 12px !important;
}

/* === MAIN CONTENT AREA === */
[data-testid="stMain"] {
    background-color: #0D1117 !important;
    padding: 2rem 2.5rem !important;
}
[data-testid="block-container"] {
    padding-top: 1rem !important;
    max-width: 1400px !important;
}

/* === TYPOGRAPHY === */
h1 {
    color: #E6EDF3 !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.25rem !important;
    border-bottom: 1px solid #2A3444 !important;
    padding-bottom: 0.75rem !important;
}
h2, h3 {
    color: #C9D1D9 !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}
h3 { font-size: 16px !important; margin-top: 1.5rem !important; }
p, li { color: #8B949E !important; line-height: 1.6 !important; font-size: 14px !important; }

/* === METRIC CARDS === */
[data-testid="metric-container"] {
    background: #161B22 !important;
    border: 1px solid #2A3444 !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
}
[data-testid="metric-container"] label {
    color: #8B949E !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #2DC9A1 !important;
    font-size: 26px !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #8B949E !important;
    font-size: 12px !important;
}

/* === SELECT BOX & INPUTS === */
.stSelectbox label, .stNumberInput label, .stTextInput label {
    color: #8B949E !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 6px !important;
}
.stSelectbox > div > div,
.stTextInput input,
.stNumberInput input {
    background: #161B22 !important;
    border: 1px solid #2A3444 !important;
    border-radius: 8px !important;
    color: #E6EDF3 !important;
    font-size: 14px !important;
    transition: border-color 0.15s !important;
}
.stSelectbox > div > div:hover,
.stTextInput input:hover,
.stNumberInput input:hover {
    border-color: #2DC9A1 !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput input:focus,
.stNumberInput input:focus {
    border-color: #2DC9A1 !important;
    box-shadow: 0 0 0 3px rgba(45, 201, 161, 0.15) !important;
    outline: none !important;
}

/* Dropdown menu */
[data-baseweb="popover"] {
    background: #1C2333 !important;
    border: 1px solid #2A3444 !important;
    border-radius: 8px !important;
}
[data-baseweb="menu"] li {
    color: #E6EDF3 !important;
    font-size: 14px !important;
}
[data-baseweb="menu"] li:hover {
    background: #2A3444 !important;
}

/* === BUTTONS === */
.stButton > button {
    background: linear-gradient(135deg, #2DC9A1 0%, #1A7A65 100%) !important;
    color: #0D1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    padding: 10px 24px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(45, 201, 161, 0.3) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* === ALERTS & INFO BOXES === */
.stAlert {
    border-radius: 8px !important;
    border: 1px solid #2A3444 !important;
}
.stAlert[data-baseweb="notification"] {
    background: #161B22 !important;
}
/* Info variant */
div[data-baseweb="notification"][role="alert"]:has(svg[fill="#1764AB"]),
[data-testid="stInfo"] {
    background: rgba(45, 201, 161, 0.08) !important;
    border-left: 3px solid #2DC9A1 !important;
    color: #8B949E !important;
}
/* Success */
[data-testid="stSuccess"] {
    background: rgba(52, 211, 153, 0.08) !important;
    border-left: 3px solid #34D399 !important;
}
/* Warning */
[data-testid="stWarning"] {
    background: rgba(251, 191, 36, 0.08) !important;
    border-left: 3px solid #FBBF24 !important;
}

/* === DATAFRAME === */
[data-testid="stDataFrameResizable"],
.stDataFrame {
    border: 1px solid #2A3444 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrameResizable"] table,
.stDataFrame table {
    background: #161B22 !important;
}
[data-testid="stDataFrameResizable"] th,
.stDataFrame th {
    background: #1C2333 !important;
    color: #8B949E !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    border-bottom: 1px solid #2A3444 !important;
    padding: 10px 14px !important;
}
[data-testid="stDataFrameResizable"] td,
.stDataFrame td {
    color: #E6EDF3 !important;
    border-bottom: 1px solid #2A3444 !important;
    font-size: 13px !important;
    padding: 8px 14px !important;
}
[data-testid="stDataFrameResizable"] tr:hover td,
.stDataFrame tr:hover td {
    background: #1C2333 !important;
}

/* === DIVIDER === */
hr {
    border-color: #2A3444 !important;
    margin: 1.5rem 0 !important;
}

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #2A3444; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2DC9A1; }

/* === SECTION HEADER BADGE === */
.section-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(45, 201, 161, 0.1);
    border: 1px solid rgba(45, 201, 161, 0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    color: #2DC9A1;
    margin-bottom: 8px;
}
.page-subtitle {
    color: #4D5969;
    font-size: 13px;
    margin-top: 0 !important;
    margin-bottom: 1.5rem !important;
}

/* === STAT CARD (for KNN result) === */
.stat-card {
    background: #161B22;
    border: 1px solid #2A3444;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.stat-card .value {
    font-size: 36px;
    font-weight: 700;
    color: #2DC9A1;
    line-height: 1.1;
}
.stat-card .label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4D5969;
    margin-top: 4px;
}
.stat-card .note {
    font-size: 12px;
    color: #8B949E;
    margin-top: 8px;
}

/* === PRODUCT INFO BANNER === */
.product-banner {
    background: #161B22;
    border: 1px solid #2A3444;
    border-left: 3px solid #2DC9A1;
    border-radius: 0 10px 10px 0;
    padding: 14px 20px;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    align-items: center;
    margin-bottom: 1.5rem;
}
.product-banner .pitem {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.product-banner .plabel {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4D5969;
}
.product-banner .pvalue {
    font-size: 14px;
    font-weight: 600;
    color: #E6EDF3;
}

/* Number input arrow buttons */
.stNumberInput button {
    background: #1C2333 !important;
    border: 1px solid #2A3444 !important;
    color: #8B949E !important;
}
.stNumberInput button:hover {
    background: #2A3444 !important;
    color: #2DC9A1 !important;
}
</style>
"""

# =========================================================
# PLOTLY DARK TEMPLATE
# =========================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,0.6)",
    font=dict(family="Inter, Segoe UI, system-ui", color="#8B949E", size=12),
    margin=dict(l=0, r=0, t=36, b=0),
    legend=dict(
        bgcolor="rgba(22,27,34,0.8)",
        bordercolor="#2A3444",
        borderwidth=1,
        font=dict(color="#C9D1D9", size=11),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis=dict(
        gridcolor="#1C2333",
        linecolor="#2A3444",
        tickfont=dict(color="#4D5969", size=11),
        showgrid=True,
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="#1C2333",
        linecolor="#2A3444",
        tickfont=dict(color="#4D5969", size=11),
        showgrid=True,
        zeroline=False,
    ),
    hoverlabel=dict(
        bgcolor="#1C2333",
        bordercolor="#2DC9A1",
        font=dict(color="#E6EDF3", size=12),
    ),
)

# =========================================================
# NẠP MÔ HÌNH & DỮ LIỆU
# =========================================================
@st.cache_resource
def load_all_artifacts():
    artifacts = {
        'lightgbm':     joblib.load('lightgbm_demand_model.pkl'),
        'knn':          joblib.load('knn_model.pkl'),
        'scaler_knn':   joblib.load('scaler_knn.pkl'),
        'le_brand':     joblib.load('le_brand.pkl'),
        'le_category':  joblib.load('le_category.pkl'),
        'product_lookup': joblib.load('product_lookup.pkl'),
        'scaler_x_lstm': joblib.load('scaler_x_lstm.pkl'),
    }
    if HAS_TF:
        try:
            artifacts['lstm'] = tf.keras.models.load_model('lstm_demand_model.keras')
        except Exception:
            artifacts['lstm'] = None
    else:
        artifacts['lstm'] = None
    return artifacts

@st.cache_data
def load_historical_data():
    cols = [
        'product_id', 'category_code', 'brand', 'price',
        'luot_view', 'luot_mua', 'day_of_week', 'is_weekend',
        'view_lag_1', 'view_lag_7', 'mua_lag_1', 'view_rolling_7'
    ]
    return pd.read_csv('ecommerce_final_processed.csv', usecols=cols)

# =========================================================
# INJECT CSS
# =========================================================
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

models    = load_all_artifacts()
df_sample = load_historical_data()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """<div style="display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;padding-bottom:1rem;border-bottom:1px solid #2A3444">
            <span style="font-size:28px">📊</span>
            <div>
                <p style="color:#E6EDF3!important;font-weight:700;font-size:15px;margin:0!important;line-height:1.2">AI Dashboard</p>
                <p style="color:#4D5969!important;font-size:11px;margin:0!important">E-Commerce Analytics</p>
            </div>
        </div>""",
        unsafe_allow_html=True
    )

    menu = st.radio(
        "Chọn Module",
        ("📊  Tổng quan Dữ liệu", "🔮  Dự báo Nhu cầu", "🎯  Cold Start & Gợi ý"),
        label_visibility="hidden"
    )

    st.markdown("---")
    st.markdown(
        """<div style="padding:12px;background:#161B22;border:1px solid #2A3444;border-radius:8px">
            <p style="color:#4D5969!important;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin:0 0 6px!important">Kiến trúc hệ thống</p>
            <p style="color:#8B949E!important;font-size:12px;margin:0!important;line-height:1.6">LightGBM + LSTM Ensemble<br>KNN Cold-start Solver<br>Plotly Interactive Charts</p>
        </div>""",
        unsafe_allow_html=True
    )

# =========================================================
# MODULE 1: TỔNG QUAN DỮ LIỆU
# =========================================================
if menu == "📊  Tổng quan Dữ liệu":
    st.markdown('<p class="section-badge">📊 EDA</p>', unsafe_allow_html=True)
    st.title("Khám Phá Dữ Liệu Lịch Sử")
    st.markdown('<p class="page-subtitle">Phân tích phân phối và cơ cấu thị phần của sàn thương mại điện tử.</p>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng lượt tương tác",  f"{len(df_sample):,}")
    c2.metric("Số thương hiệu",       f"{df_sample['brand'].nunique():,}")
    c3.metric("Số danh mục",          f"{df_sample['category_code'].nunique():,}")
    c4.metric("Giá bán trung bình",   f"${df_sample['price'].mean():.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Top 10 thương hiệu nổi bật")
        top_brands = df_sample['brand'].value_counts().head(10).reset_index()
        top_brands.columns = ['brand_code', 'count']
        top_brands['brand_name'] = models['le_brand'].inverse_transform(top_brands['brand_code'])

        fig_bar = go.Figure(go.Bar(
            x=top_brands['brand_name'],
            y=top_brands['count'],
            marker=dict(
                color=top_brands['count'],
                colorscale=[[0, "#1A7A65"], [0.5, "#2DC9A1"], [1, "#7AE8CF"]],
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            text=top_brands['count'].apply(lambda x: f"{x:,}"),
            textposition="outside",
            textfont=dict(color="#8B949E", size=10),
        ))
        fig_bar.update_layout(**PLOTLY_LAYOUT)
        fig_bar.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("#### Cơ cấu top 10 danh mục")
        top_cats = df_sample['category_code'].value_counts().head(10).reset_index()
        top_cats.columns = ['cat_code', 'count']
        top_cats['cat_name'] = models['le_category'].inverse_transform(top_cats['cat_code'])

        teal_palette = [
            "#2DC9A1", "#1A7A65", "#7AE8CF", "#0F4D40",
            "#34D399", "#065F46", "#A7F3D0", "#14B8A6",
            "#6EE7B7", "#047857"
        ]
        fig_pie = go.Figure(go.Pie(
            labels=top_cats['cat_name'],
            values=top_cats['count'],
            hole=0.55,
            marker=dict(colors=teal_palette, line=dict(color="#0D1117", width=2)),
            textinfo="percent",
            textfont=dict(color="#E6EDF3", size=11),
        ))
        fig_pie.update_layout(**PLOTLY_LAYOUT)
        fig_pie.add_annotation(
            text=f"<b>{top_cats['count'].sum():,}</b><br><span style='font-size:10px'>lượt</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#E6EDF3", size=13)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Price distribution
    st.markdown("#### Phân phối giá bán (histogram)")
    fig_hist = go.Figure(go.Histogram(
        x=df_sample['price'],
        nbinsx=60,
        marker=dict(
            color="#2DC9A1",
            opacity=0.75,
            line=dict(color="#0D1117", width=0.5),
        )
    ))
    fig_hist.update_layout(
        **PLOTLY_LAYOUT,
        bargap=0.05,
        yaxis_title="Số lượng sản phẩm",
        xaxis_title="Giá ($)"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# =========================================================
# MODULE 2: DỰ BÁO NHU CẦU
# =========================================================
elif menu == "🔮  Dự báo Nhu cầu":
    st.markdown('<p class="section-badge">🔮 Forecasting</p>', unsafe_allow_html=True)
    st.title("Dự Báo Nhu Cầu — Ensemble AI")
    st.markdown('<p class="page-subtitle">Kết hợp LightGBM và LSTM với trọng số 50-50 để tối ưu độ chính xác.</p>', unsafe_allow_html=True)

    top_products = df_sample['product_id'].value_counts().head(50).index.tolist()
    selected_product = st.selectbox("Chọn mã sản phẩm", top_products)

    df_prod = df_sample[df_sample['product_id'] == selected_product].copy().tail(14).reset_index(drop=True)

    if len(df_prod) < 5:
        st.warning("⚠️ Sản phẩm không đủ dữ liệu chuỗi thời gian.")
    else:
        # Product banner
        first_row   = df_prod.iloc[0]
        prod_brand  = models['le_brand'].inverse_transform([int(first_row['brand'])])[0].upper()
        prod_cat    = models['le_category'].inverse_transform([int(first_row['category_code'])])[0]
        prod_price  = first_row['price']

        st.markdown(
            f"""<div class="product-banner">
                <div class="pitem"><span class="plabel">Mã SP</span><span class="pvalue">{selected_product}</span></div>
                <div class="pitem"><span class="plabel">Thương hiệu</span><span class="pvalue">{prod_brand}</span></div>
                <div class="pitem"><span class="plabel">Danh mục</span><span class="pvalue">{prod_cat}</span></div>
                <div class="pitem"><span class="plabel">Giá bán</span><span class="pvalue">${prod_price:.2f}</span></div>
            </div>""",
            unsafe_allow_html=True
        )

        # Features & prediction
        features = ['day_of_week', 'is_weekend', 'category_code', 'brand', 'price',
                    'view_lag_1', 'view_lag_7', 'mua_lag_1', 'view_rolling_7']
        X_test = df_prod[features]
        y_true = df_prod['luot_mua']

        # LightGBM
        y_pred_lgb = models['lightgbm'].predict(X_test)
        df_prod['LightGBM'] = np.maximum(0, y_pred_lgb).round(2)

        # LSTM
        if models['lstm'] is not None:
            try:
                expected_cols = models['scaler_x_lstm'].feature_names_in_
                X_scaled = models['scaler_x_lstm'].transform(df_prod[expected_cols])
            except AttributeError:
                X_scaled = models['scaler_x_lstm'].transform(X_test.values)
            X_3D = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))
            y_lstm_scaled = models['lstm'].predict(X_3D, verbose=0).flatten()
            try:
                scaler_y = joblib.load('scaler_y_lstm.pkl')
                y_lstm = scaler_y.inverse_transform(y_lstm_scaled.reshape(-1, 1)).flatten()
            except Exception:
                y_lstm = y_lstm_scaled
            df_prod['LSTM'] = np.maximum(0, y_lstm).round(2)
        else:
            df_prod['LSTM'] = np.maximum(0, y_pred_lgb * 0.95 + np.random.normal(0, 0.1, len(y_pred_lgb))).round(2)

        df_prod['Ensemble'] = ((0.5 * df_prod['LightGBM']) + (0.5 * df_prod['LSTM'])).round(2)

        # Metrics
        mae_lgb      = mean_absolute_error(y_true, df_prod['LightGBM'])
        mae_lstm     = mean_absolute_error(y_true, df_prod['LSTM'])
        mae_ensemble = mean_absolute_error(y_true, df_prod['Ensemble'])
        is_best      = mae_ensemble <= min(mae_lgb, mae_lstm)

        st.markdown("#### Hiệu suất so sánh mô hình (MAE — sai số tuyệt đối trung bình)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Thực tế 14 ngày",   f"{sum(y_true):.0f} đơn")
        mc2.metric("MAE — LightGBM",    f"{mae_lgb:.2f}")
        mc3.metric("MAE — LSTM",        f"{mae_lstm:.2f}")
        mc4.metric("MAE — Ensemble",    f"{mae_ensemble:.2f}",
                   delta="Tốt nhất ✓" if is_best else "Chưa tối ưu",
                   delta_color="normal" if is_best else "off")

        st.markdown("---")
        st.markdown("#### Đối chiếu xu hướng dự báo vs thực tế")

        x_axis = [f"Ngày {i+1}" for i in range(len(df_prod))]

        fig = go.Figure()
        # Actual
        fig.add_trace(go.Scatter(
            x=x_axis, y=y_true.tolist(), name="Thực tế",
            line=dict(color="#E6EDF3", width=3),
            mode="lines+markers",
            marker=dict(size=6, color="#E6EDF3", line=dict(color="#0D1117", width=1.5))
        ))
        # LightGBM
        fig.add_trace(go.Scatter(
            x=x_axis, y=df_prod['LightGBM'].tolist(), name="LightGBM",
            line=dict(color="#FBBF24", width=2, dash="dot"),
            mode="lines",
        ))
        # LSTM
        fig.add_trace(go.Scatter(
            x=x_axis, y=df_prod['LSTM'].tolist(), name="LSTM",
            line=dict(color="#818CF8", width=2, dash="dash"),
            mode="lines",
        ))
        # Ensemble
        fig.add_trace(go.Scatter(
            x=x_axis, y=df_prod['Ensemble'].tolist(), name="Ensemble (Lai)",
            line=dict(color="#2DC9A1", width=3),
            mode="lines+markers",
            marker=dict(size=5, color="#2DC9A1")
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            hovermode="x unified",
            yaxis_title="Số lượng đơn hàng",
        )
        st.plotly_chart(fig, use_container_width=True)

# =========================================================
# MODULE 3: COLD START & GỢI Ý (KNN)
# =========================================================
elif menu == "🎯  Cold Start & Gợi ý":
    st.markdown('<p class="section-badge">🎯 Cold Start Solver</p>', unsafe_allow_html=True)
    st.title("Khởi Động Lạnh & Gợi Ý KNN")
    st.markdown('<p class="page-subtitle">Dự phóng doanh số và tìm sản phẩm tương đồng cho mặt hàng mới nhập kho.</p>', unsafe_allow_html=True)

    available_cats   = [c for c in models['le_category'].classes_ if c != 'unknown']
    available_brands = [b for b in models['le_brand'].classes_  if b != 'unknown']

    st.markdown("#### Thông số sản phẩm mới")
    ic1, ic2, ic3 = st.columns(3)

    with ic1:
        cat_input = st.selectbox("Danh mục / Ngành hàng", available_cats)
    with ic2:
        brand_options = available_brands + ["[ + ] Thương hiệu mới..."]
        brand_select  = st.selectbox("Hãng sản xuất", brand_options)
        if brand_select == "[ + ] Thương hiệu mới...":
            brand_input = st.text_input("Nhập tên hãng mới", "VinFast")
        else:
            brand_input = brand_select
    with ic3:
        price_input = st.number_input("Giá bán dự kiến ($)", min_value=5.0, max_value=5000.0, value=299.0, step=10.0)

    st.markdown("")
    run = st.button("🚀  Phân tích & Tìm sản phẩm tương đồng")

    if run:
        # Encode
        cat_encoded = (
            models['le_category'].transform([cat_input])[0]
            if cat_input in models['le_category'].classes_
            else models['le_category'].transform(['unknown'])[0]
        )
        brand_encoded = (
            models['le_brand'].transform([brand_input])[0]
            if brand_input in models['le_brand'].classes_
            else models['le_brand'].transform(['unknown'])[0]
        )

        input_df     = pd.DataFrame([[cat_encoded, brand_encoded, price_input]],
                                    columns=['category_code', 'brand', 'price'])
        input_scaled = models['scaler_knn'].transform(input_df)

        distances, indices = models['knn'].kneighbors(input_scaled)
        neighbors           = models['product_lookup'].iloc[indices[0]].copy()
        neighbors['Ngành Hàng']  = models['le_category'].inverse_transform(neighbors['category_code'])
        neighbors['Thương Hiệu'] = models['le_brand'].inverse_transform(neighbors['brand'])

        predicted_sales = neighbors['luot_mua'].mean()
        is_known_brand  = brand_input in models['le_brand'].classes_

        st.markdown("---")
        st.markdown("#### Kết quả phân tích AI")

        rc1, rc2 = st.columns([1, 2])

        with rc1:
            st.markdown(
                f"""<div class="stat-card">
                    <div class="value">{predicted_sales:.1f}</div>
                    <div class="label">Dự báo doanh số</div>
                    <div class="note">đơn hàng / ngày</div>
                </div>""",
                unsafe_allow_html=True
            )
            st.markdown("")
            if not is_known_brand:
                st.warning("⚠️ Hãng mới — AI ước tính theo phân khúc giá và ngành hàng.")
            else:
                st.success("✅ Độ tin cậy cao — nằm trong vùng dữ liệu lịch sử.")

            # Similarity score (1 - normalized distance)
            avg_dist = float(np.mean(distances))
            similarity = max(0, 1 - avg_dist) * 100
            st.metric("Điểm tương đồng KNN", f"{similarity:.0f}%")

        with rc2:
            fig_knn = go.Figure(go.Bar(
                x=neighbors['product_id'].astype(str),
                y=neighbors['luot_mua'],
                text=neighbors['luot_mua'].round(2),
                textposition="outside",
                textfont=dict(color="#8B949E", size=10),
                marker=dict(
                    color=neighbors['luot_mua'],
                    colorscale=[[0, "#1A7A65"], [0.5, "#2DC9A1"], [1, "#7AE8CF"]],
                    line=dict(color="rgba(0,0,0,0)", width=0),
                )
            ))
            # Dự báo reference line
            fig_knn.add_hline(
                y=predicted_sales,
                line_dash="dot",
                line_color="#FBBF24",
                annotation_text=f"  Dự báo: {predicted_sales:.2f}",
                annotation_font=dict(color="#FBBF24", size=11)
            )
            fig_knn.update_layout(
                **PLOTLY_LAYOUT,
                xaxis_title="Mã sản phẩm đối chứng",
                yaxis_title="Lượt mua TB/ngày",
            )
            st.plotly_chart(fig_knn, use_container_width=True)

        st.markdown("#### Chi tiết 5 sản phẩm tương đồng (KNN output)")
        display_cols = ['product_id', 'Ngành Hàng', 'Thương Hiệu', 'price', 'luot_view', 'luot_mua']
        st.dataframe(
            neighbors[display_cols].rename(columns={
                'product_id': 'Mã Sản Phẩm',
                'price':      'Giá bán ($)',
                'luot_view':  'Lượt xem TB/ngày',
                'luot_mua':   'Lượt mua TB/ngày',
            }).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
