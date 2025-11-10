"""
Dashboard Clustering Kondisi Mesin - Predictive Maintenance
Aplikasi Streamlit untuk visualisasi dan prediksi cluster kondisi mesin
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.preprocessing import StandardScaler

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Clustering Kondisi Mesin",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .cluster-info {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .program-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
        color: #000;
    }
    </style>
""", unsafe_allow_html=True)

# Data interpretasi cluster
CLUSTER_INTERPRETATIONS = {
    0: {
        "label": "Beban Tinggi Optimal",
        "torque": "40.5 - 66.7",
        "temp": "305.9 - 309.9",
        "wear": "0 - 117",
        "programs": [
            "Optimasi Jadwal Perawatan Prediktif: Tim perawatan akan menjalankan predictive maintenance dengan analisis getaran dan pengambilan sampel oli setiap 300 jam operasi pada bearing dan gearbox untuk mendeteksi pola kerusakan dini dan mencegah downtime tidak terencana.",
            "Program Penyeimbangan Beban Dinamis: Perencana produksi akan menggunakan algoritma penjadwalan untuk menjaga torsi tetap di kisaran 45–60 Nm dengan cara mendistribusikan ulang pesanan kerja setiap shift, sehingga umur komponen kritis lebih panjang dan efisiensi energi meningkat 15–20%.",
            "Peningkatan Material Komponen Beban Tinggi: Manajer perawatan akan mengganti bearing dan coupling pada shaft utama dengan baja keras dalam 2 minggu, karena torsi tinggi membutuhkan ketahanan lelah lebih baik untuk mencapai MTBF 50% lebih tinggi."
        ]
    },
    1: {
        "label": "Suhu Tinggi Ringan",
        "torque": "20.9 - 42.4",
        "temp": "309.8 - 313.7",
        "wear": "0 - 114",
        "programs": [
            "Perbaikan Sistem Pendingin Mendesak: Tim pendingin akan melakukan uji tekanan, pembersihan heat exchanger, dan penggantian coolant dalam 3 hari, karena suhu tinggi pada beban rendah menunjukkan efisiensi pendinginan menurun drastis.",
            "Pemasangan Ventilasi Tambahan: Tim fasilitas akan memasang 3 unit exhaust fan industri dengan kapasitas besar dan saluran udara optimal dalam 2 minggu untuk menurunkan suhu ruangan 5–8°C sehingga suhu proses kembali normal <310K.",
            "Pemantauan Suhu Rutin: Tim perawatan akan melakukan inspeksi pencitraan termal setiap Senin pagi sebelum shift produksi untuk mendeteksi titik panas >85°C pada motor, bearing, dan panel listrik."
        ]
    },
    2: {
        "label": "Kondisi Kritis",
        "torque": "38.9 - 66.8",
        "temp": "310 - 313.8",
        "wear": "96 - 253",
        "programs": [
            "Penghentian Darurat dan Overhaul Besar: Manajer perawatan harus menghentikan operasi mesin dalam 24 jam dan melakukan pembongkaran total selama 2 minggu dengan penggantian 80% komponen aus karena risiko kegagalan besar.",
            "Analisis Akar Masalah Komprehensif: Tim keandalan akan melakukan analisis kerusakan dengan uji metalurgi, pemeriksaan pola keausan, dan perhitungan tegangan panas untuk menemukan penyebab utama degradasi.",
            "Validasi Komisioning Bertahap: Tim kontrol kualitas akan melakukan uji jalan 72 jam dengan pemantauan lebih dari 100 parameter (torsi, suhu, getaran, keausan alat) menggunakan peningkatan beban bertahap 25–50–75–100%."
        ]
    },
    3: {
        "label": "Aus Parah",
        "torque": "20.5 - 42.8",
        "temp": "305.7 - 310.1",
        "wear": "103 - 242",
        "programs": [
            "Program Penggantian Alat Terjadwal Prioritas: Manajer ruang alat akan mengganti semua alat potong dan komponen aus dengan penggunaan >150 menit dalam 1 minggu serta membuat jadwal penggantian untuk komponen dengan penggunaan 100–150 menit.",
            "Audit Material dan Peningkatan Pelumasan: Insinyur material akan melakukan uji kekerasan dan analisis oli, lalu meningkatkan ke oli sintetis premium dan material alat dengan tingkat kekerasan lebih tinggi untuk mengurangi laju keausan 40–50%.",
            "Pelatihan Operator Parameter Optimal: Departemen pelatihan akan mengadakan workshop bulanan untuk mengajarkan kecepatan potong, laju makan, dan kedalaman potong yang tepat agar kesalahan operator tidak mempercepat keausan alat."
        ]
    },
    4: {
        "label": "Beban Tinggi Panas",
        "torque": "41.8 - 66.8",
        "temp": "309.7 - 313.6",
        "wear": "0 - 125",
        "programs": [
            "Pemasangan Chiller Kapasitas Tinggi: Insinyur pendingin akan memasang unit chiller tambahan berkapasitas 3–5 TR dengan jalur pendingin khusus dalam 3 minggu untuk menjaga stabilitas panas.",
            "Strategi Pengurangan Beban: Manajer produksi akan mendistribusikan ulang 30% beban kerja ke mesin lain mulai shift berikutnya untuk menurunkan rata-rata torsi 15–20% dan panas yang dihasilkan.",
            "Peningkatan Coolant Kinerja Tinggi: Teknisi perawatan akan melakukan pengurasan sistem coolant dan mengganti ke jenis coolant dengan konduktivitas panas lebih tinggi dan laju aliran meningkat 20%."
        ]
    },
    5: {
        "label": "Operasi Normal",
        "torque": "21.2 - 40.7",
        "temp": "305.9 - 310.1",
        "wear": "0 - 111",
        "programs": [
            "Perawatan Preventif Berkelanjutan: Teknisi level 1 akan melanjutkan perawatan rutin sesuai rekomendasi OEM setiap 1000 jam operasi atau 6 bulan dengan daftar pemeriksaan lengkap.",
            "Benchmarking dan Pemantauan Berkelanjutan: Insinyur keandalan akan menetapkan cluster ini sebagai standar kinerja dasar dengan pengumpulan data harian melalui dashboard CMMS dan tinjauan bulanan.",
            "Dokumentasi Praktik Terbaik Operator: Insinyur proses bersama operator senior akan membuat video tutorial, SOP detail, dan sesi berbagi pengetahuan setiap 3 bulan untuk transfer keahlian."
        ]
    },
    6: {
        "label": "Beban Berat Aus",
        "torque": "41.9 - 67",
        "temp": "305.7 - 310.5",
        "wear": "102 - 246",
        "programs": [
            "Penggantian Alat Darurat Prioritas 1: Manajer ruang alat harus segera mengganti semua alat dengan keausan >150 menit dalam 48 jam dan menjadwalkan penggantian untuk alat dengan keausan 100–150 menit.",
            "Penyeimbangan Beban Real-Time: Penjadwal produksi akan memindahkan pesanan kerja dengan kebutuhan torsi tinggi (>50 Nm) ke mesin dengan keausan alat <100 menit mulai shift berikutnya.",
            "Pelumasan Tingkat Lanjut dengan Sistem Otomatis: Teknisi pelumasan akan memasang sistem pelumasan otomatis dengan pelumas EP90 dan mengurangi interval siklus dari 8 jam menjadi 4 jam."
        ]
    },
    7: {
        "label": "Panas & Aus",
        "torque": "20.1 - 40",
        "temp": "309.7 - 313.7",
        "wear": "96 - 244",
        "programs": [
            "Overhaul Lengkap dan Refurbishment: Tim perawatan senior akan melakukan penghentian terjadwal dalam 1 minggu untuk pembongkaran total, pembersihan, dan perakitan ulang mesin dengan penggantian 80% komponen aus.",
            "Investigasi Akar Masalah Anomali Termal: Insinyur analisis kerusakan akan melakukan investigasi forensik menggunakan pencitraan termal, pengukuran celah bearing, pemeriksaan kebocoran seal, dan uji resistansi lilitan motor.",
            "Uji Validasi Pasca-Perbaikan: Tim jaminan kualitas bersama tim perawatan akan melakukan uji produksi berkelanjutan selama 5 hari dengan pencatatan parameter setiap 2 jam untuk memastikan overhaul berhasil."
        ]
    }
}

# Judul aplikasi
st.markdown('<div class="main-header">⚙️ Dashboard Clustering Kondisi Mesin</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistem Monitoring untuk Predictive Maintenance</div>', unsafe_allow_html=True)

# Load model dan data
@st.cache_resource
def load_models():
    """Load model KMeans dan scaler yang sudah dilatih"""
    try:
        kmeans = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return kmeans, scaler
    except FileNotFoundError:
        st.error("⚠️ Model tidak ditemukan! Pastikan file 'kmeans_model.pkl' dan 'scaler.pkl' ada di folder yang sama.")
        st.info("💡 Jalankan script 'save_model.py' terlebih dahulu untuk membuat model.")
        st.stop()

@st.cache_data
def load_data():
    """Load data clustering yang sudah ada"""
    try:
        df = pd.read_csv('clustered_data.csv')
        return df
    except FileNotFoundError:
        st.error("⚠️ Data tidak ditemukan! Pastikan file 'clustered_data.csv' ada di folder yang sama.")
        st.info("💡 Jalankan script 'save_model.py' terlebih dahulu untuk membuat data.")
        st.stop()

# Load models dan data
kmeans, scaler = load_models()
df = load_data()

# Sidebar
with st.sidebar:
    st.header("📊 Menu Navigasi")
    menu = st.radio(
        "Pilih Menu:",
        ["🏠 Overview", "📈 Visualisasi 3D", "🔮 Prediksi Cluster", "📋 Analisis Cluster"]
    )
    
    st.markdown("---")
    st.subheader("ℹ️ Informasi Model")
    st.info(f"""
    **Algoritma:** K-Means Clustering  
    **Jumlah Cluster:** {kmeans.n_clusters}  
    **Variabel:**
    - Torque [Nm]
    - Process Temperature [K]
    - Tool Wear [min]
    """)

# Menu: Overview
if menu == "🏠 Overview":
    st.header("📊 Ringkasan Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Data", f"{len(df):,}")
    with col2:
        st.metric("Jumlah Cluster", kmeans.n_clusters)
    with col3:
        st.metric("Fitur Clustering", "3")
    with col4:
        st.metric("Target Failure", f"{df['Target'].sum():,}")
    
    st.markdown("---")
    
    # Distribusi cluster
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribusi Data per Cluster")
        cluster_counts = df['Cluster'].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Jumlah Data'},
            title='Jumlah Data di Setiap Cluster',
            color=cluster_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🥧 Proporsi Cluster")
        fig = px.pie(
            values=cluster_counts.values,
            names=cluster_counts.index,
            title='Proporsi Data di Setiap Cluster',
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif per Variabel")
    features = ['Torque [Nm]', 'Process temperature [K]', 'Tool wear [min]']
    stats_df = df[features].describe().T
    st.dataframe(stats_df, use_container_width=True)
    
    st.markdown("---")
    
    # Interpretasi Cluster
    st.subheader("🏷️ Interpretasi Label dan Usulan Program per Cluster")
    st.markdown("""
    Berikut adalah interpretasi kondisi dan usulan program maintenance untuk setiap cluster 
    berdasarkan karakteristik operasional mesin.
    """)
    
    # Buat tabel interpretasi
    for cluster_id in sorted(CLUSTER_INTERPRETATIONS.keys()):
        cluster_info = CLUSTER_INTERPRETATIONS[cluster_id]
        
        with st.expander(f"**Cluster {cluster_id}: {cluster_info['label']}**", expanded=False):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**📊 Karakteristik:**")
                st.markdown(f"- **Torque:** {cluster_info['torque']} Nm")
                st.markdown(f"- **Temperature:** {cluster_info['temp']} K")
                st.markdown(f"- **Tool Wear:** {cluster_info['wear']} min")
                
                # Jumlah data di cluster
                cluster_count = len(df[df['Cluster'] == cluster_id])
                st.metric("Jumlah Mesin", f"{cluster_count:,}")
            
            with col2:
                st.markdown("**📋 Usulan Program Maintenance:**")
                for i, program in enumerate(cluster_info['programs'], 1):
                    st.markdown(f"""
                    <div class="program-box">
                    <strong>Program {i}:</strong> {program}
                    </div>
                    """, unsafe_allow_html=True)

# Menu: Visualisasi 3D
elif menu == "📈 Visualisasi 3D":
    st.header("📈 Visualisasi Clustering 3D")
    
    st.markdown("""
    Grafik 3D interaktif ini menampilkan hasil clustering berdasarkan 3 variabel utama.  
    **Cara menggunakan:**
    - 🖱️ Klik dan drag untuk memutar grafik
    - 🔍 Scroll untuk zoom in/out
    - 📌 Hover pada titik untuk melihat detail data
    """)
    
    # Opsi visualisasi
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("⚙️ Pengaturan")
        point_size = st.slider("Ukuran Titik", 1, 10, 4)
        show_centroids = st.checkbox("Tampilkan Centroid", value=True)
        opacity = st.slider("Transparansi", 0.3, 1.0, 0.8)
    
    # Buat visualisasi 3D
    fig = go.Figure()
    
    # Plot data points untuk setiap cluster
    colors = px.colors.qualitative.Set2
    for cluster in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster]
        cluster_label = CLUSTER_INTERPRETATIONS[cluster]['label']
        
        fig.add_trace(go.Scatter3d(
            x=cluster_data['Torque [Nm]'],
            y=cluster_data['Process temperature [K]'],
            z=cluster_data['Tool wear [min]'],
            mode='markers',
            name=f'Cluster {cluster}: {cluster_label}',
            marker=dict(
                size=point_size,
                color=colors[cluster % len(colors)],
                opacity=opacity,
                line=dict(width=0.5, color='white')
            ),
            hovertemplate=
                '<b>Cluster %{text}</b><br>' +
                f'<b>{cluster_label}</b><br>' +
                'Torque: %{x:.1f} Nm<br>' +
                'Process Temp: %{y:.1f} K<br>' +
                'Tool Wear: %{z:.0f} min<br>' +
                '<extra></extra>',
            text=[cluster] * len(cluster_data)
        ))
    
    # Tampilkan centroid jika diaktifkan
    if show_centroids:
        # Denormalisasi centroid
        centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
        
        fig.add_trace(go.Scatter3d(
            x=centroids_original[:, 0],
            y=centroids_original[:, 1],
            z=centroids_original[:, 2],
            mode='markers',
            name='Centroid',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond',
                line=dict(width=2, color='darkred')
            ),
            hovertemplate=
                '<b>Centroid Cluster %{text}</b><br>' +
                'Torque: %{x:.1f} Nm<br>' +
                'Process Temp: %{y:.1f} K<br>' +
                'Tool Wear: %{z:.0f} min<br>' +
                '<extra></extra>',
            text=list(range(kmeans.n_clusters))
        ))
    
    # Update layout
    fig.update_layout(
        title='Visualisasi Clustering Kondisi Mesin (3D)',
        scene=dict(
            xaxis_title='Torque [Nm]',
            yaxis_title='Process Temperature [K]',
            zaxis_title='Tool Wear [min]',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1000,
        height=700,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    with col1:
        st.plotly_chart(fig, use_container_width=True)

# Menu: Prediksi Cluster
elif menu == "🔮 Prediksi Cluster":
    st.header("🔮 Prediksi Cluster untuk Data Baru")
    
    st.markdown("""
    Masukkan nilai untuk 3 variabel di bawah ini, kemudian sistem akan memprediksi cluster yang sesuai 
    untuk kondisi mesin tersebut.
    """)
    
    # Form input
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("⚡ Torque")
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=0.1,
                help="Torsi mesin dalam Newton-meter"
            )
        
        with col2:
            st.subheader("🌡️ Process Temperature")
            proc_temp = st.number_input(
                "Process Temperature (K)",
                min_value=290.0,
                max_value=330.0,
                value=309.0,
                step=0.1,
                help="Suhu proses dalam Kelvin"
            )
        
        with col3:
            st.subheader("🔧 Tool Wear")
            tool_wear = st.number_input(
                "Tool Wear (min)",
                min_value=0,
                max_value=300,
                value=50,
                step=1,
                help="Waktu penggunaan alat dalam menit"
            )
        
        submitted = st.form_submit_button("🔍 Prediksi Cluster", use_container_width=True)
    
    if submitted:
        # Buat dataframe untuk input
        input_data = pd.DataFrame({
            'Torque [Nm]': [torque],
            'Process temperature [K]': [proc_temp],
            'Tool wear [min]': [tool_wear]
        })
        
        # Normalisasi input
        input_scaled = scaler.transform(input_data)
        
        # Prediksi cluster
        predicted_cluster = kmeans.predict(input_scaled)[0]
        
        # Hitung jarak ke setiap centroid
        distances = kmeans.transform(input_scaled)[0]
        confidence = 1 / (1 + distances[predicted_cluster])
        
        # Ambil informasi cluster
        cluster_info = CLUSTER_INTERPRETATIONS[predicted_cluster]
        
        # Tampilkan hasil
        st.markdown("---")
        st.success("✅ Prediksi Berhasil!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="cluster-info">', unsafe_allow_html=True)
            st.metric("Cluster Terprediksi", f"Cluster {predicted_cluster}", delta=None)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="cluster-info">', unsafe_allow_html=True)
            st.metric("Label Kondisi", cluster_info['label'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="cluster-info">', unsafe_allow_html=True)
            st.metric("Tingkat Kepercayaan", f"{confidence*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tampilkan usulan program
        st.subheader(f"📋 Usulan Program Maintenance untuk Cluster {predicted_cluster}")
        st.markdown(f"**Kondisi: {cluster_info['label']}**")
        
        for i, program in enumerate(cluster_info['programs'], 1):
            st.markdown(f"""
            <div class="program-box">
            <strong>Program {i}:</strong> {program}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tampilkan karakteristik cluster
        st.subheader(f"📊 Karakteristik Cluster {predicted_cluster}")
        
        cluster_data = df[df['Cluster'] == predicted_cluster][
            ['Torque [Nm]', 'Process temperature [K]', 'Tool wear [min]']
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Statistik Cluster:**")
            stats = cluster_data.describe().loc[['mean', 'std', 'min', 'max']]
            st.dataframe(stats, use_container_width=True)
        
        with col2:
            st.markdown("**Input Anda vs Rata-rata Cluster:**")
            comparison = pd.DataFrame({
                'Variabel': ['Torque', 'Process Temp', 'Tool Wear'],
                'Input Anda': [torque, proc_temp, tool_wear],
                'Rata-rata Cluster': [
                    cluster_data['Torque [Nm]'].mean(),
                    cluster_data['Process temperature [K]'].mean(),
                    cluster_data['Tool wear [min]'].mean()
                ]
            })
            st.dataframe(comparison, use_container_width=True, hide_index=True)
        
        # Visualisasi posisi input dalam cluster
        st.markdown("---")
        st.subheader("📍 Posisi Data Input dalam Cluster")
        
        fig = go.Figure()
        
        # Plot cluster data
        cluster_data_full = df[df['Cluster'] == predicted_cluster]
        fig.add_trace(go.Scatter3d(
            x=cluster_data_full['Torque [Nm]'],
            y=cluster_data_full['Process temperature [K]'],
            z=cluster_data_full['Tool wear [min]'],
            mode='markers',
            name=f'Cluster {predicted_cluster}',
            marker=dict(size=4, color='lightblue', opacity=0.6)
        ))
        
        # Plot input point
        fig.add_trace(go.Scatter3d(
            x=[torque],
            y=[proc_temp],
            z=[tool_wear],
            mode='markers',
            name='Data Input Anda',
            marker=dict(size=15, color='red', symbol='diamond')
        ))
        
        # Plot centroid
        centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
        fig.add_trace(go.Scatter3d(
            x=[centroids_original[predicted_cluster, 0]],
            y=[centroids_original[predicted_cluster, 1]],
            z=[centroids_original[predicted_cluster, 2]],
            mode='markers',
            name='Centroid',
            marker=dict(size=12, color='gold', symbol='x')
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Torque [Nm]',
                yaxis_title='Process Temperature [K]',
                zaxis_title='Tool Wear [min]'
            ),
            height=600,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Menu: Analisis Cluster
elif menu == "📋 Analisis Cluster":
    st.header("📋 Analisis Karakteristik Cluster")
    
    st.markdown("""
    Halaman ini menampilkan karakteristik detail dari setiap cluster untuk membantu 
    interpretasi hasil clustering.
    """)
    
    # Pilih cluster
    selected_cluster = st.selectbox(
        "Pilih Cluster untuk Dianalisis:",
        options=sorted(df['Cluster'].unique()),
        format_func=lambda x: f"Cluster {x}: {CLUSTER_INTERPRETATIONS[x]['label']}"
    )
    
    cluster_data = df[df['Cluster'] == selected_cluster]
    cluster_info = CLUSTER_INTERPRETATIONS[selected_cluster]
    
    st.markdown("---")
    
    # Statistik umum
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data", f"{len(cluster_data):,}")
    with col2:
        failure_rate = (cluster_data['Target'].sum() / len(cluster_data)) * 100
        st.metric("Failure Rate", f"{failure_rate:.1f}%")
    with col3:
        avg_torque = cluster_data['Torque [Nm]'].mean()
        st.metric("Avg Torque", f"{avg_torque:.1f} Nm")
    with col4:
        avg_temp = cluster_data['Process temperature [K]'].mean()
        st.metric("Avg Process Temp", f"{avg_temp:.1f} K")
    
    st.markdown("---")
    
    # Tampilkan label dan usulan program
    st.subheader(f"🏷️ {cluster_info['label']}")
    st.markdown("**📋 Usulan Program Maintenance:**")
    
    for i, program in enumerate(cluster_info['programs'], 1):
        st.markdown(f"""
        <div class="program-box">
        <strong>Program {i}:</strong> {program}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Distribusi variabel
    st.subheader(f"📊 Distribusi Variabel di Cluster {selected_cluster}")
    
    col1, col2, col3 = st.columns(3)
    
    features = ['Torque [Nm]', 'Process temperature [K]', 'Tool wear [min]']
    
    for i, (col, feature) in enumerate(zip([col1, col2, col3], features)):
        with col:
            fig = px.histogram(
                cluster_data,
                x=feature,
                nbins=30,
                title=feature.split('[')[0].strip(),
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(
                showlegend=False,
                height=300,
                margin=dict(t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Statistik detail
    st.subheader(f"📈 Statistik Detail Cluster {selected_cluster}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Statistik Deskriptif:**")
        stats = cluster_data[features].describe()
        st.dataframe(stats, use_container_width=True)
    
    with col2:
        st.markdown("**Perbandingan dengan Semua Cluster:**")
        all_stats = df[features].describe().loc['mean']
        cluster_stats = cluster_data[features].describe().loc['mean']
        comparison = pd.DataFrame({
            'Variabel': features,
            'Rata-rata Cluster': cluster_stats.values,
            'Rata-rata Keseluruhan': all_stats.values,
            'Selisih (%)': ((cluster_stats.values - all_stats.values) / all_stats.values * 100)
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Failure type distribution
    st.subheader(f"⚠️ Distribusi Jenis Failure di Cluster {selected_cluster}")
    
    failure_dist = cluster_data['Failure Type'].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            values=failure_dist.values,
            names=failure_dist.index,
            title='Proporsi Failure Type',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            x=failure_dist.index,
            y=failure_dist.values,
            title='Jumlah per Failure Type',
            labels={'x': 'Failure Type', 'y': 'Jumlah'},
            color=failure_dist.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Dashboard Clustering Kondisi Mesin - Predictive Maintenance</strong></p>
    <p>Dikembangkan untuk mendukung sistem maintenance preventif berbasis machine learning</p>
</div>
""", unsafe_allow_html=True)
