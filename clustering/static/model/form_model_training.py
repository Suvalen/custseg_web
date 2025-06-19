import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import os

# === 1. Load & clean ===
df = pd.read_csv("clustering/static/data/data_cust_final1.csv")

features = ['Age', 'Visit Frequency (per month)', 'Stay Duration (minutes)',
            'Avg Order Value (Rp)', 'Customer Rating']
df = df[features].dropna()
df = df[df >= 0].dropna()

# === 2. Preprocess ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# === 3. Clustering ===
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
kmeans.fit(X_pca)

# === 4. Save ===
output_dir = "clustering/static/model"
os.makedirs(output_dir, exist_ok=True)
joblib.dump(kmeans, f"{output_dir}/kmeans_form_model.joblib")
joblib.dump(scaler, f"{output_dir}/scaler_form.joblib")
joblib.dump(pca, f"{output_dir}/pca_form.joblib")

print("✅ Model trained and saved.")
