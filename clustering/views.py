import os
import io
import json
import pandas as pd
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
px.defaults.template = "plotly_white"
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from .models import ClusterSnapshot
from django.core.paginator import Paginator


TAG_EXPLANATIONS = {
    "High Spenders": "Customers with above-average spending per order.",
    "Low Spenders": "Customers with below-average spending per order.",
    "Moderate Spenders": "Customers with spending close to the average.",
    "Frequent Visitors": "Customers who visit the café frequently.",
    "Infrequent Visitors": "Customers who rarely visit the café.",
    "Highly Rated": "Customers who give high satisfaction ratings.",
    "Low Ratings": "Customers who give low satisfaction ratings.",
    "Extended Stays": "Customers who stay for long durations.",
    "Quick Visits": "Customers who stay briefly.",
    "Big Monthly Spenders": "Customers with high total monthly spending.",
    "Low Monthly Spenders": "Customers with low total monthly spending.",
    "Moderate Monthly Spenders": "Customers with average monthly spending.",
}
# ==== 🔧 SHARED ML FUNCTIONS ====
TAG_THRESHOLDS = {
    'avg_spend': [68000, 95000],
    'visit_freq': [5, 6],
    'rating': [2.9, 4.0],
    'stay': [30, 90],
    'total_spend': [300000, 600000],
}


REQUIRED_COLUMNS = [
    "Customer ID",
    "Avg Order Value (Rp)",
    "Visit Frequency (per month)",
    "Stay Duration (minutes)",
    "Customer Rating"
]

def clean_and_prepare_data(df):
    df.replace(9999999, pd.NA, inplace=True)
    df.dropna(inplace=True)

    for col in ['Visit Frequency (per month)', 'Stay Duration (minutes)', 'Avg Order Value (Rp)']:
        if col in df.columns:
            df = df[df[col] >= 0]
            df[col] = df[col].clip(upper=500000)

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col].astype(str))

    return df

def cluster_and_project(df, n_clusters=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_pca)

    df['Cluster'] = labels
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]

    return df

# ==== 🌐 PAGES ====

def about(request):
    info_boxes = [
        {"title": "📂 Upload Raw Data", "text": "Upload your raw customer data (CSV)."},
        {"title": "🧼 Clean Data", "text": "Removes outliers, fills missing values, and encodes categories."},
        {"title": "📊 Cluster Customers", "text": "Uses PCA and KMeans to find meaningful customer segments."},
        {"title": "📈 Visualize Clusters", "text": "Interactive scatter plot shows how customers group."},
        {"title": "📋 Explore Segments", "text": "Summarized stats per segment: spending, visits, and rating."},
        {"title": "💾 Export Insights", "text": "Download clustered results as a CSV for business use."},
    ]
    return render(request, 'about.html', {'info_boxes': info_boxes})

def home(request):
    return render(request, 'home.html')

# ==== 📂 Manual Upload ====

def upload_csv(request):
    error = None

    if request.method == 'POST':
        snapshot_name = request.POST.get('snapshot_name') or f"Manual Cluster - {pd.Timestamp.now().strftime('%B %Y')}"
        csv_file = request.FILES['file']
        df = pd.read_csv(csv_file)

        # Check for missing required columns
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            error = f"Missing required column(s): {', '.join(missing)}"
        else:
            df = clean_and_prepare_data(df)
            df = cluster_and_project(df)

            # Save to session for current session results
            request.session['clustered_data'] = df.to_json()

            # Save to database for historical viewing
            ClusterSnapshot.objects.create(
                name=snapshot_name,
                json_data=df.to_json()
            )

            return redirect('results')

    return render(request, 'upload.html', {
        'required_fields': REQUIRED_COLUMNS,
        'error': error
    })


def results(request):
    import plotly.express as px

    data_json = request.session.get('clustered_data')
    if not data_json:
        return redirect('upload')

    df = pd.read_json(io.StringIO(data_json))
    df["total_spend"] = df["Avg Order Value (Rp)"] * df["Visit Frequency (per month)"]


    # === Summary Cards ===
    cluster_summary = (
        df.groupby("Cluster")
        .agg({
            "Customer ID": "count",
            "Avg Order Value (Rp)": "mean",
            "Visit Frequency (per month)": "mean",
            "Customer Rating": "mean"
        })
        .rename(columns={"Customer ID": "count"})
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    # === Profiling Setup ===
    col_map = {
        "Avg Order Value (Rp)": "avg_spend",
        "Visit Frequency (per month)": "visit_freq",
        "Customer Rating": "rating",
        "Stay Duration (minutes)": "stay"
    }

    cluster_profiles = (
        df.groupby("Cluster")
        .agg({k: "mean" for k in col_map})
        .rename(columns=col_map)
        .round(2)
        .reset_index()
    )
    
    cluster_profiles["total_spend"] = cluster_profiles["avg_spend"] * cluster_profiles["visit_freq"]
    

    # === Quantiles for per-customer tagging ===
    
    

    def tag_customer(row):
        tags = []

        if row['Avg Order Value (Rp)'] > TAG_THRESHOLDS['avg_spend'][1]:
            tags.append("High Spenders")
        elif row['Avg Order Value (Rp)'] < TAG_THRESHOLDS['avg_spend'][0]:
            tags.append("Low Spenders")
        else:
            tags.append("Moderate Spenders")

        if row['Visit Frequency (per month)'] > TAG_THRESHOLDS['visit_freq'][1]:
            tags.append("Frequent Visitors")
        elif row['Visit Frequency (per month)'] < TAG_THRESHOLDS['visit_freq'][0]:
            tags.append("Infrequent Visitors")

        if row['Customer Rating'] >= TAG_THRESHOLDS['rating'][1]:
            tags.append("Highly Rated")
        elif row['Customer Rating'] <= TAG_THRESHOLDS['rating'][0]:
            tags.append("Low Ratings")

        if row['Stay Duration (minutes)'] > TAG_THRESHOLDS['stay'][1]:
            tags.append("Extended Stays")
        elif row['Stay Duration (minutes)'] < TAG_THRESHOLDS['stay'][0]:
            tags.append("Quick Visits")

        if row['total_spend'] > TAG_THRESHOLDS['total_spend'][1]:
            tags.append("Big Monthly Spenders")
        elif row['total_spend'] < TAG_THRESHOLDS['total_spend'][0]:
            tags.append("Low Monthly Spenders")
        else:
            tags.append("Moderate Monthly Spenders")
        return tags

    # Apply per-customer tags
    df['tags'] = df.apply(tag_customer, axis=1)

    # Format tags with HTML badges
    df['tags'] = df['tags'].apply(
        lambda tags: ' '.join([f'<span class="badge bg-secondary me-1">{tag}</span>' for tag in tags])
    )

    # === Table preview
    display_columns = [
        "Customer ID",
        "Cluster",
        "Avg Order Value (Rp)",
        "Visit Frequency (per month)",
        "Stay Duration (minutes)",
        "Customer Rating",
        "tags"
    ]
    
    profile_cards = cluster_profiles.to_dict(orient="records")
    
    paginator = Paginator(df[display_columns], 20)  # 20 rows per page
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    # === Cluster Tags for profile cards (optional)
    def tag_cluster(row):
        tags = []

        if row['avg_spend'] > TAG_THRESHOLDS['avg_spend'][1]:
            tags.append("High Spenders")
        elif row['avg_spend'] < TAG_THRESHOLDS['avg_spend'][0]:
            tags.append("Low Spenders")
        else:
            tags.append("Moderate Spenders")

        if row['visit_freq'] > TAG_THRESHOLDS['visit_freq'][1]:
            tags.append("Frequent Visitors")
        elif row['visit_freq'] < TAG_THRESHOLDS['visit_freq'][0]:
            tags.append("Infrequent Visitors")

        if row['rating'] >= TAG_THRESHOLDS['rating'][1]:
            tags.append("Highly Rated")
        elif row['rating'] <= TAG_THRESHOLDS['rating'][0]:
            tags.append("Low Ratings")

        if row['stay'] > TAG_THRESHOLDS['stay'][1]:
            tags.append("Extended Stays")
        elif row['stay'] < TAG_THRESHOLDS['stay'][0]:
            tags.append("Quick Visits")

        return tags

    cluster_profiles["tags"] = cluster_profiles.apply(tag_cluster, axis=1)
    profile_cards = cluster_profiles.to_dict(orient="records")

    # === Chart visualizations
    color_map = {
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#d62728',
    }

    count_chart = px.bar(
        pd.DataFrame(cluster_summary),
        x="Cluster", y="count", text="count",
        title="Number of Customers per Cluster",
        color="Cluster",
        color_discrete_map=color_map
    )

    spend_chart = px.bar(
        cluster_profiles, x="Cluster", y="avg_spend", text="avg_spend",
        title="Average Order Value per Cluster",
        color="Cluster", color_discrete_map=color_map
    )
    spend_chart.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    freq_chart = px.bar(
        cluster_profiles, x="Cluster", y="visit_freq", text="visit_freq",
        title="Visit Frequency per Month",
        color="Cluster", color_discrete_map=color_map
    )
    freq_chart.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    rating_chart = px.bar(
        cluster_profiles, x="Cluster", y="rating", text="rating",
        title="Average Customer Rating",
        color="Cluster", color_discrete_map=color_map
    )

    stay_chart = px.bar(
        cluster_profiles, x="Cluster", y="stay", text="stay",
        title="Average Stay Duration (Minutes)",
        color="Cluster", color_discrete_map=color_map
    )

    charts = {
        'count': count_chart.to_html(full_html=False),
        'spend': spend_chart.to_html(full_html=False),
        'frequency': freq_chart.to_html(full_html=False),
        'rating': rating_chart.to_html(full_html=False),
        'stay': stay_chart.to_html(full_html=False),
    }

    return render(request, 'results.html', {
        'summary': cluster_summary,
        'profile_cards': profile_cards,
        'charts': charts,
        'page_obj': page_obj,
        'tag_explanations': TAG_EXPLANATIONS,
    })





# ==== 📊 Plotly View ====

def plot(request):
    data_json = request.session.get('clustered_data')
    if not data_json:
        return redirect('upload')

    df = pd.read_json(io.StringIO(data_json))
    fig = px.scatter(
        df, x="PC1", y="PC2", color="Cluster",
        title="Customer Segmentation Clusters",
        hover_data=df.columns
    )
    plot_html = fig.to_html(full_html=False)
    return render(request, 'plot.html', {'plot_html': plot_html})

# ==== 🧼 Separate Cleaning Page ====

def clean_data(request):
    error = None

    if request.method == 'POST':
        raw_file = request.FILES['file']
        df = pd.read_csv(raw_file)

        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            error = f"Missing required column(s): {', '.join(missing)}"
        else:
            request.session['raw_data_len'] = df.shape[0]
            df = clean_and_prepare_data(df)
            request.session['cleaned_data'] = df.to_json()
            return redirect('cleaned_results')

    return render(request, 'clean_data.html', {
        'required_fields': REQUIRED_COLUMNS,
        'error': error
    })


def cleaned_results(request):
    data_json = request.session.get('cleaned_data')
    if not data_json:
        return redirect('clean_data')
    df = pd.read_json(io.StringIO(data_json))
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    encoded_count = len(df.select_dtypes(include='int64').columns)
    preview_html = df.head(20).to_html(classes='table table-striped table-sm', index=False)
    original_len = request.session.get('raw_data_len', total_rows)
    rows_removed = original_len - total_rows

    return render(request, 'cleaned_results.html', {
        'total_rows': total_rows,
        'total_cols': total_cols,
        'encoded_count': encoded_count,
        'rows_removed': rows_removed,
        'preview': preview_html,
    })

def upload_cleaned(request):
    data_json = request.session.get('cleaned_data')
    if not data_json:
        return redirect('clean_data')
    df = pd.read_json(io.StringIO(data_json))
    df = cluster_and_project(df)
    request.session['clustered_data'] = df.to_json()
    return redirect('results')

# ==== 🧩 Internal Clustering Page ====

def internal_cluster(request):
    # Load internal data file
    file_path = os.path.join(settings.BASE_DIR, 'clustering', 'static', 'data', 'customer_data.csv')
    df = pd.read_csv(file_path)

    # Clean + Cluster
    df = clean_and_prepare_data(df)
    df = cluster_and_project(df)

    # === Profiling Logic ===
    col_map = {
        "Avg Order Value (Rp)": "avg_spend",
        "Visit Frequency (per month)": "visit_freq",
        "Customer Rating": "rating",
        "Stay Duration (minutes)": "stay"
    }

    cluster_profiles = (
        df.groupby("Cluster")
        .agg({k: "mean" for k in col_map})
        .rename(columns=col_map)
        .round(2)
        .reset_index()
    )

    spend_q = cluster_profiles['avg_spend'].quantile([0.33, 0.66]).values
    visit_q = cluster_profiles['visit_freq'].quantile([0.33, 0.66]).values
    rating_q = cluster_profiles['rating'].quantile([0.33, 0.66]).values
    stay_q = cluster_profiles['stay'].quantile([0.33, 0.66]).values

def tag_cluster(row):
    tags = []

    if row['avg_spend'] > TAG_THRESHOLDS['avg_spend'][1]:
        tags.append("High Spenders")
    elif row['avg_spend'] < TAG_THRESHOLDS['avg_spend'][0]:
        tags.append("Low Spenders")
    else:
        tags.append("Moderate Spenders")

    if row['visit_freq'] > TAG_THRESHOLDS['visit_freq'][1]:
        tags.append("Frequent Visitors")
    elif row['visit_freq'] < TAG_THRESHOLDS['visit_freq'][0]:
        tags.append("Infrequent Visitors")

    if row['rating'] >= TAG_THRESHOLDS['rating'][1]:
        tags.append("Highly Rated")
    elif row['rating'] <= TAG_THRESHOLDS['rating'][0]:
        tags.append("Low Ratings")

    if row['stay'] > TAG_THRESHOLDS['stay'][1]:
        tags.append("Extended Stays")
    elif row['stay'] < TAG_THRESHOLDS['stay'][0]:
        tags.append("Quick Visits")

    return tags


    cluster_profiles["tags"] = cluster_profiles.apply(tag_cluster, axis=1)
    
    import plotly.express as px

    color_map = {
    0: '#1f77b4',  # blue
    1: '#ff7f0e',  # orange
    2: '#2ca02c',  # green
    3: '#d62728',  # red
}

# Count per cluster
    count_chart = px.bar(
    df.groupby("Cluster").size().reset_index(name="count"),
    x="Cluster", y="count", text="count",
    title="Customers per Cluster",
    color="Cluster", color_discrete_map=color_map
)

# Avg spend
    spend_chart = px.bar(
    cluster_profiles, x="Cluster", y="avg_spend", text="avg_spend",
    title="Average Order Value", color="Cluster", color_discrete_map=color_map
)

# Visit frequency
    freq_chart = px.bar(
    cluster_profiles, x="Cluster", y="visit_freq", text="visit_freq",
    title="Visit Frequency", color="Cluster", color_discrete_map=color_map
)

# Rating
    rating_chart = px.bar(
    cluster_profiles, x="Cluster", y="rating", text="rating",
    title="Customer Rating", color="Cluster", color_discrete_map=color_map
)

# Stay duration
    stay_chart = px.bar(
    cluster_profiles, x="Cluster", y="stay", text="stay",
    title="Stay Duration", color="Cluster", color_discrete_map=color_map
)

# Apply styling (optional helper function)
    for chart in [count_chart, spend_chart, freq_chart, rating_chart, stay_chart]:
        chart.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
        marker_line_color='white',
        marker_line_width=1.5
    )
    chart.update_layout(template="plotly_white", margin=dict(t=40, b=20, l=0, r=0))

# Render HTML
    charts = {
    'count': count_chart.to_html(full_html=False),
    'spend': spend_chart.to_html(full_html=False),
    'frequency': freq_chart.to_html(full_html=False),
    'rating': rating_chart.to_html(full_html=False),
    'stay': stay_chart.to_html(full_html=False),
}


    # Merge tags into the main DataFrame
    df = df.merge(cluster_profiles[["Cluster", "tags"]], on="Cluster", how="left")

    # === Table: only selected columns, limit to first 20 rows
    display_columns = [
        "Customer ID",
        "Cluster",
        "Avg Order Value (Rp)",
        "Visit Frequency (per month)",
        "Stay Duration (minutes)",
        "Customer Rating",
        "tags"
    ]

    table_html = df[display_columns].head(20).to_html(
        classes='table table-striped table-bordered table-sm text-center',
        index=False
    )

    # === Summary cards
    summary = (
        df.groupby("Cluster")
        .agg({
            "Customer ID": "count",
            "Avg Order Value (Rp)": "mean",
            "Visit Frequency (per month)": "mean",
            "Customer Rating": "mean"
        })
        .rename(columns={
            "Customer ID": "count",
            "Avg Order Value (Rp)": "avg_spend",
            "Visit Frequency (per month)": "visit_freq",
            "Customer Rating": "rating"
        })
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    profile_cards = cluster_profiles.to_dict(orient="records")

    return render(request, 'internal_cluster.html', {
    'summary': summary,
    'clustered_data': table_html,
    'profile_cards': profile_cards,
    'charts': charts
})



# ==== 📥 Export ====

def export_clustered_csv(request):
    data_json = request.session.get('clustered_data')
    if not data_json:
        return redirect('results')

    df = pd.read_json(io.StringIO(data_json))
    df["total_spend"] = df["Avg Order Value (Rp)"] * df["Visit Frequency (per month)"]

    # === Recreate plain tags for export ===
    def tag_customer(row):
        tags = []

        if row['Avg Order Value (Rp)'] > TAG_THRESHOLDS['avg_spend'][1]:
            tags.append("High Spenders")
        elif row['Avg Order Value (Rp)'] < TAG_THRESHOLDS['avg_spend'][0]:
            tags.append("Low Spenders")
        else:
            tags.append("Moderate Spenders")

        if row['Visit Frequency (per month)'] > TAG_THRESHOLDS['visit_freq'][1]:
            tags.append("Frequent Visitors")
        elif row['Visit Frequency (per month)'] < TAG_THRESHOLDS['visit_freq'][0]:
            tags.append("Infrequent Visitors")

        if row['Customer Rating'] >= TAG_THRESHOLDS['rating'][1]:
            tags.append("Highly Rated")
        elif row['Customer Rating'] <= TAG_THRESHOLDS['rating'][0]:
            tags.append("Low Ratings")

        if row['Stay Duration (minutes)'] > TAG_THRESHOLDS['stay'][1]:
            tags.append("Extended Stays")
        elif row['Stay Duration (minutes)'] < TAG_THRESHOLDS['stay'][0]:
            tags.append("Quick Visits")

        if row['total_spend'] > TAG_THRESHOLDS['total_spend'][1]:
            tags.append("Big Monthly Spenders")
        elif row['total_spend'] < TAG_THRESHOLDS['total_spend'][0]:
            tags.append("Low Monthly Spenders")
        else:
            tags.append("Moderate Monthly Spenders")

        return ', '.join(tags)

    df['tags'] = df.apply(tag_customer, axis=1)

    # Select only the columns you want to export
    export_columns = [
        "Customer ID",
        "Cluster",
        "Avg Order Value (Rp)",
        "Visit Frequency (per month)",
        "Stay Duration (minutes)",
        "Customer Rating",
        "tags"
    ]
    export_df = df[export_columns]

    # Generate CSV response
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=clustered_results.csv'
    export_df.to_csv(response, index=False)

    return response

def export_cleaned_csv(request):
    data_json = request.session.get('cleaned_data')
    if not data_json:
        return redirect('clean_data')
    df = pd.read_json(io.StringIO(data_json))
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename=cleaned_data.csv'
    df.to_csv(response, index=False)
    return response


def cluster_detail(request, cluster_id):
    return render(request, 'cluster.html', {'cluster_id': cluster_id})

def plot_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"

def demographics_report(request):
    df = pd.read_csv(os.path.join(settings.BASE_DIR, 'clustering/static/data/data_cust_final1.csv'))

    gender_counts = df['Gender'].value_counts()
    work_counts = df['Work Type'].value_counts()
    age_data = df['Age'].dropna()

    # Gender Plot
    fig1, ax1 = plt.subplots()
    sns.barplot(x=gender_counts.index, y=gender_counts.values, ax=ax1)
    ax1.set_title("Gender Distribution")
    gender_plot = plot_to_html(fig1)

    # Age Plot
    fig2, ax2 = plt.subplots()
    sns.histplot(age_data, bins=10, kde=True, ax=ax2)
    ax2.set_title("Age Distribution")
    age_plot = plot_to_html(fig2)

    # Work Type Plot
    fig3, ax3 = plt.subplots()
    sns.barplot(x=work_counts.index, y=work_counts.values, ax=ax3)
    ax3.set_title("Work Type Distribution")
    ax3.tick_params(axis='x', rotation=45)
    work_plot = plot_to_html(fig3)

    return render(request, 'demographics.html', {
        'gender_plot': gender_plot,
        'age_plot': age_plot,
        'work_plot': work_plot
    })

# views.py
import os
import numpy as np
import joblib
from django.shortcuts import render
from django.conf import settings

MODEL_DIR = os.path.join(settings.BASE_DIR, 'clustering', 'static', 'model')
model = joblib.load(os.path.join(MODEL_DIR, 'kmeans_form_model.joblib'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_form.joblib'))
pca = joblib.load(os.path.join(MODEL_DIR, 'pca_form.joblib'))

def form_predict(request):
    if request.method == 'POST':
        try:
            # === 1) Read the form ===
            age         = float(request.POST['age'])
            freq        = float(request.POST['freq'])
            stay        = float(request.POST['stay'])
            order_value = float(request.POST['order_value'])
            rating      = float(request.POST['rating'])

            # === 1b) Define your feature columns ===
            feature_columns = [
                'Age',
                'Visit Frequency (per month)',
                'Stay Duration (minutes)',
                'Avg Order Value (Rp)',
                'Customer Rating'
            ]

            # === 2) Load & clean your full reference data ===
            path = os.path.join(settings.BASE_DIR,
                                'clustering/static/data/data_cust_final1.csv')
            df_ref = pd.read_csv(path)
            df_ref.replace(9999999, pd.NA, inplace=True)
            df_ref.dropna(inplace=True)
            for c in [
                'Visit Frequency (per month)',
                'Stay Duration (minutes)',
                'Avg Order Value (Rp)'
            ]:
                df_ref = df_ref[df_ref[c] >= 0]
                df_ref[c] = df_ref[c].clip(upper=500000)

            # === 3) Compute the manual cluster profiles exactly as in results() ===
            #   – scale, PCA & predict on the entire reference set
            ref_scaled = scaler.transform(df_ref[feature_columns])
            ref_pca    = pca.transform(ref_scaled)
            df_ref['Cluster'] = model.predict(ref_pca)

            # build cluster_profiles with mean stats + total_spend
            col_map = {
                "Avg Order Value (Rp)":        "avg_spend",
                "Visit Frequency (per month)": "visit_freq",
                "Customer Rating":             "rating",
                "Stay Duration (minutes)":     "stay"
            }
            cluster_profiles = (
                df_ref
                  .groupby('Cluster')
                  .agg({k: 'mean' for k in col_map})
                  .rename(columns=col_map)
                  .round(2)
                  .reset_index()
            )
            cluster_profiles['total_spend'] = (
                cluster_profiles['avg_spend'] * cluster_profiles['visit_freq']
            )

            # === 4) Compute quantile thresholds for tagging ===
            def q33_66(s): return s.quantile([.33, .66]).values
            TAG_THRESHOLDS = {
                'avg_spend':   q33_66(cluster_profiles['avg_spend']),
                'visit_freq':  q33_66(cluster_profiles['visit_freq']),
                'rating':      q33_66(cluster_profiles['rating']),
                'stay':        q33_66(cluster_profiles['stay']),
                'total_spend': q33_66(cluster_profiles['total_spend']),
            }

            # === 5) Build the single‐row user vector for profiling & tagging ===
            user_row = {
                'avg_spend':   order_value,
                'visit_freq':  freq,
                'rating':      rating,
                'stay':        stay,
                'total_spend': order_value * freq
            }

            # === 6) Assign to nearest cluster‐profile by Euclidean distance ===
            profile_feats = ['avg_spend','visit_freq','rating','stay','total_spend']
            prof_matrix = cluster_profiles[profile_feats].to_numpy()
            user_vector = np.array([user_row[f] for f in profile_feats])
            dists = np.linalg.norm(prof_matrix - user_vector, axis=1)
            best_idx = dists.argmin()
            user_cluster = int(cluster_profiles.loc[best_idx, 'Cluster'])

            # === 7) Tag the user row with the same thresholds ===
            def tag(vals):
                tags = []
                for feat, (lo, hi), names in [
                    ('avg_spend',   TAG_THRESHOLDS['avg_spend'],
                     ("Low Spenders","Moderate Spenders","High Spenders")),
                    ('visit_freq',  TAG_THRESHOLDS['visit_freq'],
                     ("Infrequent Visitors",None,"Frequent Visitors")),
                    ('rating',      TAG_THRESHOLDS['rating'],
                     ("Low Ratings",None,"Highly Rated")),
                    ('stay',        TAG_THRESHOLDS['stay'],
                     ("Quick Visits",None,"Extended Stays")),
                    ('total_spend', TAG_THRESHOLDS['total_spend'],
                     ("Low Monthly Spenders","Moderate Monthly Spenders","Big Monthly Spenders")),
                ]:
                    v = vals[feat]
                    if v < lo:
                        tags.append(names[0])
                    elif v > hi:
                        tags.append(names[2])
                    else:
                        if names[1]:
                            tags.append(names[1])
                return tags

            user_tags = tag(user_row)

            # === 8) Render the result ===
            segment_map = {
                0: "Cluster 0 – …",
                1: "Cluster 1 – …",
                2: "Cluster 2 – …",
            }
            return render(request, 'form_result.html', {
                'cluster': user_cluster,
                'segment': segment_map.get(user_cluster, f"Cluster {user_cluster}"),
                'tags':    user_tags
            })

        except Exception as e:
            return render(request, 'form_predict.html', {'error': str(e)})

    return render(request, 'form_predict.html')

def snapshot_list(request):
    snapshots = ClusterSnapshot.objects.order_by('-created_at')
    for s in snapshots:
        print(f"Snapshot: {s.name}, ID: {s.id}")
    return render(request, 'snapshot_list.html', {'snapshots': snapshots})
    
from django.shortcuts import render, get_object_or_404
from .models import ClusterSnapshot
import pandas as pd
import io
import plotly.express as px

def snapshot_detail(request, snapshot_id):
    snapshot = get_object_or_404(ClusterSnapshot, id=snapshot_id)
    df = pd.read_json(io.StringIO(snapshot.json_data))

    # === Summary Cards ===
    snapshot = get_object_or_404(ClusterSnapshot, id=snapshot_id)
    df = pd.read_json(io.StringIO(snapshot.json_data))  # <-- already the snapshot data!


    df = pd.read_json(io.StringIO(snapshot.json_data))
    df["total_spend"] = df["Avg Order Value (Rp)"] * df["Visit Frequency (per month)"]


    # === Summary Cards ===
    cluster_summary = (
        df.groupby("Cluster")
        .agg({
            "Customer ID": "count",
            "Avg Order Value (Rp)": "mean",
            "Visit Frequency (per month)": "mean",
            "Customer Rating": "mean"
        })
        .rename(columns={"Customer ID": "count"})
        .round(2)
        .reset_index()
        .to_dict(orient="records")
    )

    # === Profiling Setup ===
    col_map = {
        "Avg Order Value (Rp)": "avg_spend",
        "Visit Frequency (per month)": "visit_freq",
        "Customer Rating": "rating",
        "Stay Duration (minutes)": "stay"
    }

    cluster_profiles = (
        df.groupby("Cluster")
        .agg({k: "mean" for k in col_map})
        .rename(columns=col_map)
        .round(2)
        .reset_index()
    )
    
    cluster_profiles["total_spend"] = cluster_profiles["avg_spend"] * cluster_profiles["visit_freq"]
    

    # === Quantiles for per-customer tagging ===
    
    

    def tag_customer(row):
        tags = []

        if row['Avg Order Value (Rp)'] > TAG_THRESHOLDS['avg_spend'][1]:
            tags.append("High Spenders")
        elif row['Avg Order Value (Rp)'] < TAG_THRESHOLDS['avg_spend'][0]:
            tags.append("Low Spenders")
        else:
            tags.append("Moderate Spenders")

        if row['Visit Frequency (per month)'] > TAG_THRESHOLDS['visit_freq'][1]:
            tags.append("Frequent Visitors")
        elif row['Visit Frequency (per month)'] < TAG_THRESHOLDS['visit_freq'][0]:
            tags.append("Infrequent Visitors")

        if row['Customer Rating'] >= TAG_THRESHOLDS['rating'][1]:
            tags.append("Highly Rated")
        elif row['Customer Rating'] <= TAG_THRESHOLDS['rating'][0]:
            tags.append("Low Ratings")

        if row['Stay Duration (minutes)'] > TAG_THRESHOLDS['stay'][1]:
            tags.append("Extended Stays")
        elif row['Stay Duration (minutes)'] < TAG_THRESHOLDS['stay'][0]:
            tags.append("Quick Visits")

        if row['total_spend'] > TAG_THRESHOLDS['total_spend'][1]:
            tags.append("Big Monthly Spenders")
        elif row['total_spend'] < TAG_THRESHOLDS['total_spend'][0]:
            tags.append("Low Monthly Spenders")
        else:
            tags.append("Moderate Monthly Spenders")
        return tags

    # Apply per-customer tags
    df['tags'] = df.apply(tag_customer, axis=1)

    # Format tags with HTML badges
    df['tags'] = df['tags'].apply(
        lambda tags: ' '.join([f'<span class="badge bg-secondary me-1">{tag}</span>' for tag in tags])
    )

    # === Table preview
    display_columns = [
        "Customer ID",
        "Cluster",
        "Avg Order Value (Rp)",
        "Visit Frequency (per month)",
        "Stay Duration (minutes)",
        "Customer Rating",
        "tags"
    ]
    profile_cards = cluster_profiles.to_dict(orient="records")
    
    paginator = Paginator(df[display_columns], 20)  # 20 rows per page
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    # === Cluster Tags for profile cards (optional)
    def tag_cluster(row):
        tags = []

        if row['avg_spend'] > TAG_THRESHOLDS['avg_spend'][1]:
            tags.append("High Spenders")
        elif row['avg_spend'] < TAG_THRESHOLDS['avg_spend'][0]:
            tags.append("Low Spenders")
        else:
            tags.append("Moderate Spenders")

        if row['visit_freq'] > TAG_THRESHOLDS['visit_freq'][1]:
            tags.append("Frequent Visitors")
        elif row['visit_freq'] < TAG_THRESHOLDS['visit_freq'][0]:
            tags.append("Infrequent Visitors")

        if row['rating'] >= TAG_THRESHOLDS['rating'][1]:
            tags.append("Highly Rated")
        elif row['rating'] <= TAG_THRESHOLDS['rating'][0]:
            tags.append("Low Ratings")

        if row['stay'] > TAG_THRESHOLDS['stay'][1]:
            tags.append("Extended Stays")
        elif row['stay'] < TAG_THRESHOLDS['stay'][0]:
            tags.append("Quick Visits")

        return tags

    cluster_profiles["tags"] = cluster_profiles.apply(tag_cluster, axis=1)
    profile_cards = cluster_profiles.to_dict(orient="records")

    # === Chart visualizations
    color_map = {
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#d62728',
    }

    count_chart = px.bar(
        pd.DataFrame(cluster_summary),
        x="Cluster", y="count", text="count",
        title="Number of Customers per Cluster",
        color="Cluster",
        color_discrete_map=color_map
    )

    spend_chart = px.bar(
        cluster_profiles, x="Cluster", y="avg_spend", text="avg_spend",
        title="Average Order Value per Cluster",
        color="Cluster", color_discrete_map=color_map
    )
    spend_chart.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    freq_chart = px.bar(
        cluster_profiles, x="Cluster", y="visit_freq", text="visit_freq",
        title="Visit Frequency per Month",
        color="Cluster", color_discrete_map=color_map
    )
    freq_chart.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    rating_chart = px.bar(
        cluster_profiles, x="Cluster", y="rating", text="rating",
        title="Average Customer Rating",
        color="Cluster", color_discrete_map=color_map
    )

    stay_chart = px.bar(
        cluster_profiles, x="Cluster", y="stay", text="stay",
        title="Average Stay Duration (Minutes)",
        color="Cluster", color_discrete_map=color_map
    )

    charts = {
        'count': count_chart.to_html(full_html=False),
        'spend': spend_chart.to_html(full_html=False),
        'frequency': freq_chart.to_html(full_html=False),
        'rating': rating_chart.to_html(full_html=False),
        'stay': stay_chart.to_html(full_html=False),
    }

    return render(request, 'results.html', {
        'page_obj': page_obj,
        'summary': cluster_summary,
        'profile_cards': profile_cards,
        'charts': charts
    })
    
from django.contrib import messages

def delete_snapshot(request, pk):
    snapshot = get_object_or_404(ClusterSnapshot, pk=pk)
    snapshot.delete()
    messages.success(request, f"Snapshot '{snapshot.name}' deleted successfully.")
    return redirect('snapshot_list')

def export_snapshot_csv(request, snapshot_id):
    snapshot = get_object_or_404(ClusterSnapshot, id=snapshot_id)
    df = pd.read_json(io.StringIO(snapshot.json_data))
    df["total_spend"] = df["Avg Order Value (Rp)"] * df["Visit Frequency (per month)"]

    def tag_customer(row):
        tags = []

        if row['Avg Order Value (Rp)'] > TAG_THRESHOLDS['avg_spend'][1]:
            tags.append("High Spenders")
        elif row['Avg Order Value (Rp)'] < TAG_THRESHOLDS['avg_spend'][0]:
            tags.append("Low Spenders")
        else:
            tags.append("Moderate Spenders")

        if row['Visit Frequency (per month)'] > TAG_THRESHOLDS['visit_freq'][1]:
            tags.append("Frequent Visitors")
        elif row['Visit Frequency (per month)'] < TAG_THRESHOLDS['visit_freq'][0]:
            tags.append("Infrequent Visitors")

        if row['Customer Rating'] >= TAG_THRESHOLDS['rating'][1]:
            tags.append("Highly Rated")
        elif row['Customer Rating'] <= TAG_THRESHOLDS['rating'][0]:
            tags.append("Low Ratings")

        if row['Stay Duration (minutes)'] > TAG_THRESHOLDS['stay'][1]:
            tags.append("Extended Stays")
        elif row['Stay Duration (minutes)'] < TAG_THRESHOLDS['stay'][0]:
            tags.append("Quick Visits")

        if row['total_spend'] > TAG_THRESHOLDS['total_spend'][1]:
            tags.append("Big Monthly Spenders")
        elif row['total_spend'] < TAG_THRESHOLDS['total_spend'][0]:
            tags.append("Low Monthly Spenders")
        else:
            tags.append("Moderate Monthly Spenders")

        return ', '.join(tags)

    df['tags'] = df.apply(tag_customer, axis=1)

    export_df = df[[
        "Customer ID", "Cluster", "Avg Order Value (Rp)",
        "Visit Frequency (per month)", "Stay Duration (minutes)",
        "Customer Rating", "tags"
    ]]

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename=snapshot_{snapshot.id}_clusters.csv'
    export_df.to_csv(response, index=False)

    return response