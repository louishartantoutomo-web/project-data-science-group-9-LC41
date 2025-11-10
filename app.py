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
    </style>
""", unsafe_allow_html=True)

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
    - Rotational Speed
    - Torque
    - Tool Wear
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
    features = ['Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    stats_df = df[features].describe().T
    st.dataframe(stats_df, use_container_width=True)

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
        
        fig.add_trace(go.Scatter3d(
            x=cluster_data['Rotational speed [rpm]'],
            y=cluster_data['Torque [Nm]'],
            z=cluster_data['Tool wear [min]'],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(
                size=point_size,
                color=colors[cluster % len(colors)],
                opacity=opacity,
                line=dict(width=0.5, color='white')
            ),
            hovertemplate=
                '<b>Cluster %{text}</b><br>' +
                'Rotational Speed: %{x:.1f} rpm<br>' +
                'Torque: %{y:.1f} Nm<br>' +
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
                'Rotational Speed: %{x:.1f} rpm<br>' +
                'Torque: %{y:.1f} Nm<br>' +
                'Tool Wear: %{z:.0f} min<br>' +
                '<extra></extra>',
            text=list(range(kmeans.n_clusters))
        ))
    
    # Update layout
    fig.update_layout(
        title='Visualisasi Clustering Kondisi Mesin (3D)',
        scene=dict(
            xaxis_title='Rotational Speed [rpm]',
            yaxis_title='Torque [Nm]',
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
            st.subheader("🔄 Rotational Speed")
            rot_speed = st.number_input(
                "Rotational Speed (rpm)",
                min_value=0,
                max_value=5000,
                value=1500,
                step=10,
                help="Kecepatan rotasi mesin dalam RPM"
            )
        
        with col2:
            st.subheader("⚡ Torque")
            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=100.0,
                value=40.0,
                step=0.1,
                help="Torsi mesin dalam Newton-meter"
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
            'Rotational speed [rpm]': [rot_speed],
            'Torque [Nm]': [torque],
            'Tool wear [min]': [tool_wear]
        })
        
        # Normalisasi input
        input_scaled = scaler.transform(input_data)
        
        # Prediksi cluster
        predicted_cluster = kmeans.predict(input_scaled)[0]
        
        # Hitung jarak ke setiap centroid
        distances = kmeans.transform(input_scaled)[0]
        confidence = 1 / (1 + distances[predicted_cluster])
        
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
            st.metric("Tingkat Kepercayaan", f"{confidence*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            cluster_size = len(df[df['Cluster'] == predicted_cluster])
            st.markdown('<div class="cluster-info">', unsafe_allow_html=True)
            st.metric("Ukuran Cluster", f"{cluster_size:,} data")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Tampilkan karakteristik cluster
        st.subheader(f"📊 Karakteristik Cluster {predicted_cluster}")
        
        cluster_data = df[df['Cluster'] == predicted_cluster][
            ['Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Statistik Cluster:**")
            stats = cluster_data.describe().loc[['mean', 'std', 'min', 'max']]
            st.dataframe(stats, use_container_width=True)
        
        with col2:
            st.markdown("**Input Anda vs Rata-rata Cluster:**")
            comparison = pd.DataFrame({
                'Variabel': ['Rotational Speed', 'Torque', 'Tool Wear'],
                'Input Anda': [rot_speed, torque, tool_wear],
                'Rata-rata Cluster': [
                    cluster_data['Rotational speed [rpm]'].mean(),
                    cluster_data['Torque [Nm]'].mean(),
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
            x=cluster_data_full['Rotational speed [rpm]'],
            y=cluster_data_full['Torque [Nm]'],
            z=cluster_data_full['Tool wear [min]'],
            mode='markers',
            name=f'Cluster {predicted_cluster}',
            marker=dict(size=4, color='lightblue', opacity=0.6)
        ))
        
        # Plot input point
        fig.add_trace(go.Scatter3d(
            x=[rot_speed],
            y=[torque],
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
                xaxis_title='Rotational Speed [rpm]',
                yaxis_title='Torque [Nm]',
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
        format_func=lambda x: f"Cluster {x}"
    )
    
    cluster_data = df[df['Cluster'] == selected_cluster]
    
    st.markdown("---")
    
    # Statistik umum
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data", f"{len(cluster_data):,}")
    with col2:
        failure_rate = (cluster_data['Target'].sum() / len(cluster_data)) * 100
        st.metric("Failure Rate", f"{failure_rate:.1f}%")
    with col3:
        avg_rot = cluster_data['Rotational speed [rpm]'].mean()
        st.metric("Avg Rot. Speed", f"{avg_rot:.0f} rpm")
    with col4:
        avg_torque = cluster_data['Torque [Nm]'].mean()
        st.metric("Avg Torque", f"{avg_torque:.1f} Nm")
    
    st.markdown("---")
    
    # Distribusi variabel
    st.subheader(f"📊 Distribusi Variabel di Cluster {selected_cluster}")
    
    col1, col2, col3 = st.columns(3)
    
    features = ['Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    
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
