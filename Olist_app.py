import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide"
)

DATA_PATH = "Olist_Dataset_Clustering.csv"
MODEL_PATH = "rfm_kmeans_pipeline.pkl"

FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "payment_installments",
    "price",
    "review_score"
]

# =========================
# CSS
# =========================
CUSTOM_CSS = """
<style>
/* =========================
    BACKGROUND & BASE TEXT
========================= */
.stApp {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
    color: #FFFFFF !important;
    font-family: 'Inter', ui-sans-serif, system-ui;
}

/* =========================
    SIDEBAR
========================= */
section[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.2) !important;
    backdrop-filter: blur(15px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* =========================
    CARDS
========================= */
.card {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    margin-bottom: 20px;
}

/* =========================
    TEXT COLORS
========================= */
h1, h2, h3, h4, h5, p, span, label, .stMarkdown {
    color: #FFFFFF !important;
}

/* Sub-text atau caption */
.stCaption, small {
    color: rgba(255, 255, 255, 0.7) !important;
}

/* =========================
    INPUT FIELDS
========================= */
/* Number Input & Selectbox */
div[data-baseweb="select"] > div, 
div[data-testid="stNumberInput"] input {
    background-color: rgba(255, 255, 255, 0.9) !important;
    color: #1e3c72 !important; 
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* =========================
    BUTTONS
========================= */
div.stButton > button {
    background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 0.6rem 1.5rem !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3) !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 242, 254, 0.5) !important;
    color: #FFFFFF !important;
}

/* Sidebar Menu Buttons */
section[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    box-shadow: none !important;
    text-transform: none;
}

.menu-active button {
    background: rgba(255, 255, 255, 0.2) !important;
    border: 1px solid #4facfe !important;
}

/* =========================
    TABLE
========================= */
.table-scroll {
    width: 100% !important;
    overflow-x: auto !important;
    overflow-y: hidden !important;
    border-radius: 16px !important;
    border: 2px solid rgba(79, 172, 254, 0.4) !important; 
    background: rgba(255, 255, 255, 0.95) !important;
    box-shadow: 0 12px 24px rgba(0,0,0,0.3) !important;
    margin-bottom: 20px;
}

.table-scroll table {
    width: max-content !important;
    min-width: 100% !important;
    border-collapse: collapse !important;
    font-size: 14px !important;
}

/* Header Tabel */
.table-scroll thead th {
    background: #1e3c72 !important; 
    color: #FFFFFF !important;
    font-weight: 950 !important;
    text-align: left !important;
    padding: 12px 10px !important;
    border-bottom: 3px solid #4facfe !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    white-space: nowrap !important;
}

/* Body Tabel */
.table-scroll tbody td {
    padding: 10px 10px !important;
    border-bottom: 1px solid rgba(0, 0, 0, 0.05) !important;
    border-right: 1px solid rgba(0, 0, 0, 0.05) !important;
    color: #071F45 !important; 
    font-weight: 600 !important;
    white-space: nowrap !important;
}

/* Efek Hover Baris Tabel */
.table-scroll tbody tr:hover {
    background: rgba(79, 172, 254, 0.1) !important;
}

/* =========================
    SELECTBOX & INPUT FIX
========================= */
div[data-baseweb="select"] span, 
div[data-baseweb="select"] input,
div[data-testid="stNumberInput"] input {
    color: #1e3c72 !important;
    font-weight: 800 !important;
}

ul[role="listbox"] li {
    color: #1e3c72 !important;
    background-color: #FFFFFF !important;
}

div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #4facfe !important;
}

/* DIVIDER */
hr {
    border: none !important;
    height: 3px !important;
    background-color: rgba(255, 255, 255, 0.8) !important;
    margin: 30px 0 !important;
    border-radius: 10px !important;
}

/* =========================
    HIDE HEADER & FOOTER
========================= */
header {
    visibility: hidden;
    height: 0% !important;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Adjust top padding */
.stAppViewMain > section > div {
    padding-top: 2rem !important;
}
</style>
"""

# Menerapkan CSS ke Streamlit
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# FUNCTION
# =========================
def title_case_cols(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        c2 = str(c).replace("_", " ").strip()
        c2 = " ".join([w.capitalize() for w in c2.split()])
        new_cols.append(c2)
    df2 = df.copy()
    df2.columns = new_cols
    return df2

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2 = df2.replace([np.inf, -np.inf], np.nan)
    df2 = df2.dropna(how="all")
    df2 = df2.dropna(axis=1, how="all")
    return df2

def df_to_html_table(df: pd.DataFrame, max_rows=50):
    df2 = clean_df(df).head(max_rows)
    df2 = title_case_cols(df2)
    return f"""
    <div class="table-scroll">
        {df2.to_html(index=False, escape=False)}
    </div>
    """

def plot_template():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF", size=13),
        margin=dict(l=50, r=20, t=60, b=50),
        title=dict(font=dict(color="#FFFFFF", size=22, weight="bold"), x=0, y=0.95),
        legend=dict(
            font=dict(color="#FFFFFF", size=12),
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="#FFFFFF",
            borderwidth=1,
            orientation="v",
            itemclick="toggleothers"
        ),
        xaxis=dict(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)', 
            linecolor='#FFFFFF', 
            tickfont=dict(color="#FFFFFF")
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='rgba(255,255,255,0.1)', 
            linecolor='#FFFFFF', 
            tickfont=dict(color="#FFFFFF")
        )
    )

def title_case_col(c: str) -> str:
    c2 = str(c).replace("_", " ").strip()
    return " ".join([w.capitalize() for w in c2.split()])

def make_download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def pretty_table(df: pd.DataFrame, max_rows=50):
    st.markdown(df_to_html_table(df, max_rows=max_rows), unsafe_allow_html=True)

def validate_manual_input(recency, frequency, monetary, payment_installments, price, review_score):
    errors = []
    if recency < 0: errors.append("Recency tidak boleh negatif.")
    if frequency <= 0: errors.append("Frequency minimal 1.")
    if monetary < 0: errors.append("Monetary tidak boleh negatif.")
    if payment_installments < 0: errors.append("Payment Installments tidak boleh negatif.")
    if price < 0: errors.append("Price tidak boleh negatif.")
    if review_score < 1 or review_score > 5: errors.append("Review Score harus antara 1 sampai 5.")
    return errors

def soft_divider():
    st.markdown("<div class='soft-line'></div>", unsafe_allow_html=True)

# =========================
# LOAD DATA + MODEL
# =========================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# Error handling jika file tidak ada
try:
    df = load_data(DATA_PATH)
    pipeline = load_model(MODEL_PATH)
except:
    st.warning("Pastikan file dataset dan model tersedia di direktori.")

# =========================
# SIDEBAR MENU
# =========================
if "menu" not in st.session_state:
    st.session_state.menu = "Home"

def menu_btn(label, key):
    active = (st.session_state.menu == key)
    cls = "menu-active" if active else ""
    st.markdown(f"<div class='{cls}'>", unsafe_allow_html=True)
    clicked = st.sidebar.button(label, key=f"btn_{key}", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if clicked:
        st.session_state.menu = key
        st.rerun()

st.sidebar.markdown("<div class='menu-title'>📌 Menu</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='menu-wrap'>", unsafe_allow_html=True)
menu_btn("🏠  Home", "Home")
menu_btn("📋  Data Preview & Statistik", "Data Preview & Statistik")
menu_btn("📊 Exploratory Data Analysis", "EDA")
menu_btn("🧠  Prediksi Cluster", "Prediksi Cluster")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

# =========================
# MAIN CONTENT
# =========================
menu = st.session_state.menu

if "Home" in menu:
    st.title("🛍️ Customer Segmentation Dashboard")
    st.markdown("""
    <div class="card">
        <h3>Selamat Datang!</h3>
        <p>Dashboard ini dirancang sebagai solusi analitik untuk mengatasi tantangan rendahnya tingkat pembelian ulang (repeat buying) dan retentation rate pada platform e-commerce Olist dengan menggunakan algoritma K-Means Clustering. Sistem ini akan memetakan perilaku pelanggan secara mendalam melalui metrik RFM (Recency, Frequency, Monetary) dan indikator lain seperti Payment Installments, Price, dan Review Score.</p>
    </div>
    """, unsafe_allow_html=True)

elif menu == "Data Preview & Statistik":
    st.subheader("📋 Data Preview")
    n = st.slider("Jumlah Baris Preview", 5, 100, 10, step=5)
    pretty_table(df, max_rows=n)
    st.markdown("---")
    st.subheader("📌 Statistik Deskriptif")
    stats = df[FEATURES].describe().T.reset_index().rename(columns={"index": "Feature"})
    pretty_table(stats)

elif menu == "EDA":
    st.subheader("📊 Exploratory Data Analysis (EDA)")

    # 1. Payment Method
    if "payment_type" in df.columns:
        tmp = df["payment_type"].value_counts().reset_index()
        tmp.columns = ["Payment Type", "Count"]
        
        fig = px.pie(tmp, names="Payment Type", values="Count", 
                     title="Payment Method Distribution", 
                     hole=0.4)
        
        fig.update_traces(
            textposition='outside', 
            textinfo='percent+label',
            marker=dict(line=dict(color='#FFFFFF', width=1))
        )
        
        fig.update_layout(**plot_template())
        
        fig.update_layout(
            legend_font_color="#FFFFFF",
            legend_title_font_color="#FFFFFF"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")

    # 2. Top 5 Customer States
    if "customer_state" in df.columns:
        tmp = df["customer_state"].value_counts().head(5).reset_index()
        tmp.columns = ["Customer State", "Total Orders"]
        fig = px.bar(tmp, x="Customer State", y="Total Orders", 
                     title="Top 5 Customer State by Orders", color="Total Orders", 
                     color_continuous_scale="Blues")
        fig.update_layout(**plot_template())
        fig.update_coloraxes(colorbar_tickfont_color="#FFFFFF", colorbar_title_font_color="#FFFFFF")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 3. Top 10 Product Categories
    if "product_category_name_english" in df.columns:
        # 1. Mengambil top 10 category
        tmp = df["product_category_name_english"].value_counts().head(10).reset_index()
        tmp.columns = ["Product Category", "Total Orders"]
        
        # 2. Mengurutkan dataframe dari terkecil ke terbesar 
        tmp = tmp.sort_values(by="Total Orders", ascending=True) 

        fig = px.bar(tmp, x="Total Orders", y="Product Category", 
                     orientation='h', 
                     title="Top 10 Product Categories by Orders", 
                     color="Total Orders", 
                     color_continuous_scale="Viridis")
        
        fig.update_layout(**plot_template())
        fig.update_coloraxes(colorbar_tickfont_color="#FFFFFF", colorbar_title_font_color="#FFFFFF")
        
        fig.update_xaxes(linecolor='#FFFFFF', showline=True)
        fig.update_yaxes(linecolor='#FFFFFF', showline=True, categoryorder='total ascending')
        
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 4. Histogram Feature
    st.subheader("Histogram Feature")
    feature = st.selectbox("Pilih Feature untuk Melihat Distribusinya", FEATURES, index=0)
    
    fig = px.histogram(df, x=feature, nbins=40, 
                       title=f"Distribusi {feature.replace('_',' ').title()}",
                       color_discrete_sequence=["#4facfe"])
    
    fig.update_layout(**plot_template())
    
    fig.update_xaxes(title_text=feature.replace('_', ' ').title(), linecolor='#FFFFFF', showline=True, mirror=True)
    fig.update_yaxes(title_text="Count", linecolor='#FFFFFF', showline=True, mirror=True)
    
    st.plotly_chart(fig, use_container_width=True)

elif menu == "Prediksi Cluster":
    st.subheader("🎯 Prediksi Cluster")
    st.caption("Pilih manual input atau upload CSV untuk memprediksi cluster customer.")
    st.write("")

    tab1, tab2 = st.tabs(["🧮 Manual Input", "📤 Upload CSV"])

    with tab1:
        # Layout manual input
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            recency = st.number_input("Recency", 0, 100000, 200)
            frequency = st.number_input("Frequency", 1, 100000, 1)
        with c2:
            monetary = st.number_input("Monetary", 0.0, 1e9, 120.0)
            payment_installments = st.number_input("Payment Installments", 0, 1000, 2)
        with c3:
            price = st.number_input("Price", 0.0, 1e9, 80.0)
            review_score = st.number_input("Review Score", 1.0, 5.0, 5.0)

        if st.button("Predict Cluster", use_container_width=True):
            errors = validate_manual_input(recency, frequency, monetary, payment_installments, price, review_score)
            if errors: 
                st.error("\n".join(errors))
            else:
                # 1. Menyiapkan data untuk prediksi
                X = pd.DataFrame([{
                    "recency": recency, 
                    "frequency": frequency, 
                    "monetary": monetary, 
                    "payment_installments": payment_installments, 
                    "price": price, 
                    "review_score": review_score
                }])
                
                # 2. Melakukan Prediksi
                cluster = pipeline.predict(X)[0]
                
                # 3. Menampilkan Hasil
                st.markdown(f'''
                    <div class="card" style="
                        max-width: 800px; 
                        margin: 20px auto; 
                        padding: 2px !important; 
                        min-height: 50px;
                        text-align: center;
                        border: 2px solid #4facfe;">
                        <h1 style="color:#FFFFFF; margin:0; font-size: 2rem;">
                            ✅ Hasil Prediksi: <span style="color:#00f2fe;">Cluster {cluster}</span>
                        </h1>
                    </div>
                ''', unsafe_allow_html=True)

                # 4. Menambahkan keterangan di bawah hasil prediksi
                st.write("")
                
                if cluster == 0:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border-left: 5px solid #4facfe;">
                        <h4 style="margin-top:0; color:#4facfe;">📋 Segment: Satisfied & Low-Spend Buyers</h4>
                        <p><b>Karakteristik:</b> Pelanggan dengan nilai transaksi kecil, harga produk rendah, penggunaan cicilan rendah, namun memiliki review score tinggi. Didominasi pembelian produk kategori kebutuhan rumah.</p>
                        <p style="margin-bottom: 5px;"><b>Strategi Retention:</b></p>
                        <ul style="margin-bottom:0;">
                            <li>Loyalty points untuk pembelian berulang.</li>
                            <li>Gratis ongkir dengan minimum spend rendah-menengah.</li>
                            <li>Reminder campaign untuk repeat purchase.</li>
                        </ul>
                        <p style="margin-top: 20px; margin-bottom: 5px;"><b>Strategi Campaign:</b></p>
                        <ul style="margin-bottom:0;">
                            <li>Bundle produk kebutuhan rumah.</li>
                            <li>Cross-selling produk pelengkap rumah tangga.</li>
                            <li>Flash sale ringan untuk dorong impulse buying.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                elif cluster == 1:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border-left: 5px solid #ff4b4b;">
                        <h4 style="margin-top:0; color:#ff4b4b;">📋 Segment: High-Spend At-Risk Buyers</h4>
                        <p><b>Karakteristik:</b> High spender dengan frekuensi rendah, review score rendah, dan dominasi kategori furniture kantor. Segmen bernilai tinggi namun berisiko churn.</p>
                        <p style="margin-bottom: 5px;"><b>Strategi Retention:</b></p>
                        <ul style="margin-bottom:0;">
                            <li>Priority customer service.</li>
                            <li>Proactive follow-up setelah pembelian besar.</li>
                            <li>Service recovery untuk review rendah.</li>
                        </ul>
                        <p style="margin-top: 20px; margin-bottom: 5px;"><b>Strategi Campaign:</b></p>
                        <ul style="margin-bottom:0;">
                            <li>Penawaran eksklusif furniture kantor.</li>
                            <li>Extended warranty / free installation.</li>
                            <li>Voucher kompensasi untuk pengalaman negatif.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                elif cluster == 2:
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border-left: 5px solid #faca2e;">
                        <h4 style="margin-top:0; color:#faca2e;">📋 Segment: Premium Installment Buyers</h4>
                        <p><b>Karakteristik:</b> Pembeli produk premium dengan harga tinggi dan penggunaan cicilan tinggi. Frekuensi rendah, nilai transaksi besar, dan review score cukup tinggi namun bervariasi.</p>
                        <p style="margin-bottom: 5px;"><b>Strategi Retention:</b></p>
                        <ul style="margin-bottom:0;">
                            <li>VIP customer program.</li>
                            <li>Dedicated customer support.</li>
                            <li>Reminder untuk upgrade/premium products.</li>
                        </ul>
                        <p style="margin-top: 20px; margin-bottom: 5px;"><b>Strategi Campaign:</b></p>
                        <ul style="margin-bottom:0;">
                            <li>Promo cicilan 0% / extended installment.</li>
                            <li>Early access produk premium & gift edition.</li>
                            <li>Personalized premium recommendations.</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

    with tab2:
        st.subheader("📤 Prediksi Cluster dari File CSV")
        with st.expander(" File CSV harus memiliki nama kolom berikut (case-sensitive):", expanded=True):
            st.markdown("""
            - **Wajib:** recency, frequency, monetary, payment_installment, price, dan review_score
            - **Opsional:** customer_unique_id, product_category_name_english, payment_type, customer_city
            """)        
        uploaded = st.file_uploader("Upload CSV File", type=["csv"])

        # Inisialisasi Session State agar data tidak hilang saat slider digeser
        if "df_hasil_prediksi" not in st.session_state:
            st.session_state.df_hasil_prediksi = None
        if "last_uploaded_file" not in st.session_state:
            st.session_state.last_uploaded_file = None

        if uploaded is not None:
            # Jika user upload file baru, reset hasil prediksi lama
            if st.session_state.last_uploaded_file != uploaded.name:
                st.session_state.df_hasil_prediksi = None
                st.session_state.last_uploaded_file = uploaded.name

            try:
                up_df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Gagal membaca file CSV. Error: {e}")
                st.stop()

            st.subheader("🔍 Preview File Upload (5 Baris)")
            st.write(f"Rows: **{up_df.shape[0]:,}** | Cols: **{up_df.shape[1]:,}**")
            pretty_table(up_df, max_rows=5)

            # Validasi Kolom
            missing = [c for c in FEATURES if c not in up_df.columns]
            if missing:
                st.error(f"File tidak memiliki kolom: {', '.join([col.capitalize() for col in missing])}")
            else:
                # Tombol Run
                run = st.button("Run Prediction", use_container_width=True)

                if run:
                    with st.spinner("Sedang memproses cluster..."):
                        try:
                            X_input = up_df[FEATURES].copy()
                            preds = pipeline.predict(X_input)
                            
                            # Simpan hasil ke session state
                            result_df = up_df.copy()
                            result_df["cluster"] = preds
                            st.session_state.df_hasil_prediksi = result_df
                        except Exception as e:
                            st.error(f"Gagal prediksi: {e}")

                # Menampilkan hasil jika sudah ada di Session State
                if st.session_state.df_hasil_prediksi is not None:
                    res_df = st.session_state.df_hasil_prediksi
                    
                    st.markdown("---")
                    st.success("Prediksi selesai ✅")
                    
                    # Mapping Kolom
                    column_mapping = {
                        "customer_unique_id": "Customer ID",
                        "recency": "Recency",
                        "frequency": "Frequency",
                        "monetary": "Monetary",
                        "payment_installments": "Payment Installments",
                        "price": "Price",
                        "review_score": "Review Score",
                        "product_category_name_english": "Product Category",
                        "payment_type": "Payment Method",
                        "customer_city": "Customer City",
                        "cluster": "Cluster"
                    }

                    # Filter kolom yang ada
                    cols_to_select = [c for c in column_mapping.keys() if c in res_df.columns]
                    final_view = res_df[cols_to_select].rename(columns=column_mapping).fillna("-")

                    # Slider
                    num_show = st.slider("Pilih Jumlah Baris yang Ingin Ditampilkan", 5, 100, 10, step=5, key="slider_predict_csv")
                    
                    st.subheader("📌 Hasil Prediksi")
                    pretty_table(final_view, max_rows=num_show)

                    st.write("")
                    csv_bytes = make_download_csv(final_view)
                    st.download_button(
                        "⬇️ Download Hasil CSV",
                        data=csv_bytes,
                        file_name="olist_cluster_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.info("Silakan Upload File CSV untuk Mulai Prediksi.")
            st.session_state.df_hasil_prediksi = None