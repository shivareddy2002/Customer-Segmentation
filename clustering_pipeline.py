"""
============================================================
  Amazon Customer Segmentation - ML Pipeline
  Full RFM + Behavioral Feature Engineering
  K-Means | Hierarchical | DBSCAN
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid", palette="muted")
PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  AMAZON CUSTOMER SEGMENTATION PIPELINE")
print("=" * 55)

df = pd.read_csv("data/amazon_sales_clustering_dataset.csv")
print(f"\n✔ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head(3))

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
print("\n[STEP 2] Data Cleaning...")
print("Missing values:\n", df.isnull().sum())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Convert date
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# Remove outlier prices / quantities
df = df[(df['price'] > 0) & (df['quantity'] > 0)]

print(f"✔ Clean dataset: {df.shape[0]} rows")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[STEP 3] Feature Engineering...")

df['total_spent']     = df['price'] * df['quantity']
df['purchase_month']  = df['purchase_date'].dt.month
df['purchase_dayofweek'] = df['purchase_date'].dt.dayofweek

# ─────────────────────────────────────────────
# 4. RFM MODEL
# ─────────────────────────────────────────────
print("\n[STEP 4] Building RFM Features...")

snapshot_date = df['purchase_date'].max() + pd.Timedelta(days=1)

rfm = df.groupby('customer_id').agg({
    'purchase_date': lambda x: (snapshot_date - x.max()).days,
    'product_id'   : 'count',
    'total_spent'  : 'sum'
}).reset_index()

rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']

print(rfm.describe())

# ─────────────────────────────────────────────
# 5. BEHAVIORAL FEATURES
# ─────────────────────────────────────────────
print("\n[STEP 5] Behavioral Feature Engineering...")

behavior = df.groupby('customer_id').agg(
    total_quantity    = ('quantity',         'sum'),
    avg_price         = ('price',            'mean'),
    avg_rating        = ('rating',           'mean'),
    total_reviews     = ('review_count',     'sum'),
    unique_categories = ('product_category', 'nunique'),
    avg_order_value   = ('total_spent',      'mean'),
    max_single_spend  = ('total_spent',      'max'),
    purchase_days     = ('purchase_date',    'nunique'),
    weekend_purchases = ('purchase_dayofweek', lambda x: (x >= 5).sum()),
).reset_index()

# ─────────────────────────────────────────────
# 6. MERGE & FINAL CUSTOMER TABLE
# ─────────────────────────────────────────────
customer_df = pd.merge(rfm, behavior, on='customer_id')

# Derived engagement scores
customer_df['engagement_score'] = (
    customer_df['avg_rating'] * np.log1p(customer_df['total_reviews'])
)
customer_df['loyalty_index'] = (
    customer_df['Frequency'] / (customer_df['Recency'] + 1)
)

print(f"\n✔ Customer features built: {customer_df.shape}")
print(customer_df.describe())

# ─────────────────────────────────────────────
# 7. SCALING
# ─────────────────────────────────────────────
print("\n[STEP 6] Feature Scaling...")

feature_cols = [c for c in customer_df.columns if c != 'customer_id']
features = customer_df[feature_cols]

scaler = StandardScaler()
scaled = scaler.fit_transform(features)
scaled_df = pd.DataFrame(scaled, columns=feature_cols)

joblib.dump(scaler, "models/scaler.pkl")
print("✔ Scaler saved.")

# ─────────────────────────────────────────────
# 8. OPTIMAL CLUSTERS - ELBOW + SILHOUETTE
# ─────────────────────────────────────────────
print("\n[STEP 7] Finding Optimal Clusters...")

wcss       = []
sil_scores = []
k_range    = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(scaled_df)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(scaled_df, lbl))
    print(f"  k={k}  WCSS={km.inertia_:.1f}  Silhouette={sil_scores[-1]:.4f}")

best_k = k_range.start + np.argmax(sil_scores)
print(f"\n✔ Best k = {best_k}  (Silhouette = {max(sil_scores):.4f})")

# Save elbow plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(list(k_range), wcss, 'o-', color='#E94560', lw=2)
axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Number of Clusters')
axes[0].set_ylabel('WCSS')
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), sil_scores, 's-', color='#0F3460', lw=2)
axes[1].set_title('Silhouette Score', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('Score')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/elbow_silhouette.png", dpi=120, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 9. KMEANS FINAL MODEL
# ─────────────────────────────────────────────
print(f"\n[STEP 8] Training K-Means (k={best_k})...")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=15, max_iter=500)
customer_df['cluster'] = kmeans.fit_predict(scaled_df)

joblib.dump(kmeans,  "models/kmeans_model.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("✔ K-Means model saved.")

final_score = silhouette_score(scaled_df, customer_df['cluster'])
print(f"✔ Final Silhouette Score: {final_score:.4f}")

# ─────────────────────────────────────────────
# 10. HIERARCHICAL CLUSTERING DENDROGRAM
# ─────────────────────────────────────────────
print("\n[STEP 9] Hierarchical Clustering...")

sample = scaled_df.sample(min(300, len(scaled_df)), random_state=42)
linked = linkage(sample, method='ward')

plt.figure(figsize=(14, 6))
dendrogram(linked, truncate_mode='lastp', p=20,
           leaf_rotation=45, leaf_font_size=10,
           color_threshold=None)
plt.title("Hierarchical Clustering Dendrogram (Ward)", fontsize=14, fontweight='bold')
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/dendrogram.png", dpi=120, bbox_inches='tight')
plt.close()
print("✔ Dendrogram saved.")

# ─────────────────────────────────────────────
# 11. DBSCAN
# ─────────────────────────────────────────────
print("\n[STEP 10] DBSCAN Clustering...")

dbscan = DBSCAN(eps=1.5, min_samples=5)
customer_df['dbscan_cluster'] = dbscan.fit_predict(scaled_df)
n_noise = (customer_df['dbscan_cluster'] == -1).sum()
print(f"  DBSCAN clusters found: {customer_df['dbscan_cluster'].nunique() - 1}")
print(f"  Noise points: {n_noise}")

# ─────────────────────────────────────────────
# 12. PCA VISUALIZATION
# ─────────────────────────────────────────────
print("\n[STEP 11] PCA 2D Visualization...")

pca = PCA(n_components=2)
pca_coords = pca.fit_transform(scaled_df)
pca_df = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'])
pca_df['cluster'] = customer_df['cluster']

explained = pca.explained_variance_ratio_
print(f"  Variance explained: PC1={explained[0]*100:.1f}%  PC2={explained[1]*100:.1f}%")

joblib.dump(pca, "models/pca.pkl")

palette = {0: '#E94560', 1: '#0F3460', 2: '#16213E', 3: '#533483',
           4: '#05C4B6', 5: '#F5A623'}

plt.figure(figsize=(10, 7))
for cl in sorted(pca_df['cluster'].unique()):
    mask = pca_df['cluster'] == cl
    plt.scatter(pca_df.loc[mask,'PC1'], pca_df.loc[mask,'PC2'],
                s=40, alpha=0.7, label=f"Cluster {cl}",
                c=palette.get(cl, '#888888'))

plt.title(f"Customer Segments — PCA\n(Variance: {sum(explained)*100:.1f}%)",
          fontsize=14, fontweight='bold')
plt.xlabel(f"PC1 ({explained[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}%)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/pca_clusters.png", dpi=120, bbox_inches='tight')
plt.close()

# ─────────────────────────────────────────────
# 13. DISTRIBUTION PLOTS (10 GRAPHS)
# ─────────────────────────────────────────────
print("\n[STEP 12] Generating 10 Visualization Plots...")

def save_fig(name):
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{name}.png", dpi=120, bbox_inches='tight')
    plt.close()

# Plot 1 – Recency Distribution
plt.figure(figsize=(8, 5))
sns.histplot(customer_df['Recency'], bins=30, color='#E94560', kde=True)
plt.title("Recency Distribution (Days Since Last Purchase)", fontweight='bold')
plt.xlabel("Days"); plt.ylabel("Count")
save_fig("recency_dist")

# Plot 2 – Frequency Distribution
plt.figure(figsize=(8, 5))
sns.histplot(customer_df['Frequency'], bins=30, color='#0F3460', kde=True)
plt.title("Purchase Frequency Distribution", fontweight='bold')
plt.xlabel("Purchases"); plt.ylabel("Count")
save_fig("frequency_dist")

# Plot 3 – Monetary Distribution
plt.figure(figsize=(8, 5))
sns.histplot(customer_df['Monetary'], bins=30, color='#533483', kde=True)
plt.title("Total Spending Distribution", fontweight='bold')
plt.xlabel("Total Spent ($)"); plt.ylabel("Count")
save_fig("monetary_dist")

# Plot 4 – Cluster Size
plt.figure(figsize=(8, 5))
cluster_counts = customer_df['cluster'].value_counts().sort_index()
bars = plt.bar(cluster_counts.index.astype(str),
               cluster_counts.values,
               color=['#E94560','#0F3460','#533483','#16213E','#05C4B6'])
for bar, val in zip(bars, cluster_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             str(val), ha='center', fontsize=11, fontweight='bold')
plt.title("Customer Count per Segment", fontweight='bold')
plt.xlabel("Cluster"); plt.ylabel("Customers")
save_fig("cluster_count")

# Plot 5 – Recency vs Monetary Scatter
plt.figure(figsize=(9, 6))
for cl in sorted(customer_df['cluster'].unique()):
    sub = customer_df[customer_df['cluster'] == cl]
    plt.scatter(sub['Recency'], sub['Monetary'], alpha=0.6, s=35,
                label=f"Cluster {cl}", c=palette.get(cl, '#888'))
plt.title("Recency vs Total Spending by Segment", fontweight='bold')
plt.xlabel("Recency (days)"); plt.ylabel("Monetary ($)")
plt.legend()
save_fig("recency_monetary")

# Plot 6 – Frequency vs Monetary
plt.figure(figsize=(9, 6))
for cl in sorted(customer_df['cluster'].unique()):
    sub = customer_df[customer_df['cluster'] == cl]
    plt.scatter(sub['Frequency'], sub['Monetary'], alpha=0.6, s=35,
                label=f"Cluster {cl}", c=palette.get(cl, '#888'))
plt.title("Purchase Frequency vs Spending by Segment", fontweight='bold')
plt.xlabel("Frequency"); plt.ylabel("Monetary ($)")
plt.legend()
save_fig("freq_monetary")

# Plot 7 – Rating by Cluster
plt.figure(figsize=(9, 5))
sns.boxplot(x='cluster', y='avg_rating', data=customer_df,
            palette=['#E94560','#0F3460','#533483','#16213E'])
plt.title("Average Rating by Segment", fontweight='bold')
plt.xlabel("Cluster"); plt.ylabel("Avg Rating")
save_fig("rating_cluster")

# Plot 8 – Spending by Cluster
plt.figure(figsize=(9, 5))
sns.boxplot(x='cluster', y='Monetary', data=customer_df,
            palette=['#E94560','#0F3460','#533483','#16213E'])
plt.title("Total Spending Distribution by Segment", fontweight='bold')
plt.xlabel("Cluster"); plt.ylabel("Monetary ($)")
save_fig("spending_cluster")

# Plot 9 – Loyalty Index by Cluster
plt.figure(figsize=(9, 5))
sns.violinplot(x='cluster', y='loyalty_index', data=customer_df,
               palette=['#E94560','#0F3460','#533483','#16213E'], inner='box')
plt.title("Loyalty Index by Segment", fontweight='bold')
plt.xlabel("Cluster"); plt.ylabel("Loyalty Index")
save_fig("loyalty_cluster")

# Plot 10 – Correlation Heatmap
plt.figure(figsize=(13, 9))
corr_cols = ['Recency','Frequency','Monetary','avg_rating',
             'unique_categories','engagement_score','loyalty_index','cluster']
corr = customer_df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap='RdYlBu_r', vmin=-1, vmax=1,
            linewidths=0.5, square=True)
plt.title("Feature Correlation Heatmap", fontweight='bold', fontsize=13)
save_fig("correlation_heatmap")

print("✔ All 10 plots saved.")

# ─────────────────────────────────────────────
# 14. CLUSTER ANALYSIS SUMMARY
# ─────────────────────────────────────────────
print("\n[STEP 13] Cluster Summary...")

summary_cols = ['cluster','Recency','Frequency','Monetary',
                'avg_rating','unique_categories','loyalty_index','engagement_score']
cluster_summary = customer_df[summary_cols].groupby('cluster').mean().round(2)
cluster_summary['customer_count'] = customer_df.groupby('cluster').size()
cluster_summary.to_csv("outputs/cluster_summary.csv")
print(cluster_summary)

# ─────────────────────────────────────────────
# 15. LABEL CLUSTER SEGMENTS
# ─────────────────────────────────────────────
def label_cluster(row):
    # Sort by Monetary desc to identify high-value
    if row['Monetary'] == cluster_summary['Monetary'].max():
        return "High-Value Customers"
    elif row['Frequency'] == cluster_summary['Frequency'].max():
        return "Frequent Buyers"
    elif row['Recency'] == cluster_summary['Recency'].max():
        return "Occasional Buyers"
    else:
        return "Discount Shoppers"

cluster_labels = {}
max_monetary = cluster_summary['Monetary'].idxmax()
max_frequency = cluster_summary['Frequency'].idxmax()
max_recency   = cluster_summary['Recency'].idxmax()

for cl in cluster_summary.index:
    if cl == max_monetary:
        cluster_labels[cl] = "High-Value Customers"
    elif cl == max_frequency:
        cluster_labels[cl] = "Frequent Buyers"
    elif cl == max_recency:
        cluster_labels[cl] = "Occasional Buyers"
    else:
        cluster_labels[cl] = "Discount Shoppers"

joblib.dump(cluster_labels, "models/cluster_labels.pkl")
print(f"\n✔ Cluster Labels: {cluster_labels}")

customer_df['segment_name'] = customer_df['cluster'].map(cluster_labels)

# ─────────────────────────────────────────────
# 16. RECOMMENDATION ENGINE
# ─────────────────────────────────────────────
print("\n[STEP 14] Building Recommendation Engine...")

raw_df = pd.read_csv("data/amazon_sales_clustering_dataset.csv")
raw_df['purchase_date'] = pd.to_datetime(raw_df['purchase_date'])

merged = raw_df.merge(customer_df[['customer_id','cluster','segment_name']],
                      on='customer_id')

top_categories = (
    merged.groupby(['cluster','product_category'])
          .size()
          .reset_index(name='count')
          .sort_values(['cluster','count'], ascending=[True, False])
          .groupby('cluster')
          .head(3)
)
top_categories.to_csv("outputs/category_recommendations.csv", index=False)
print(top_categories)

# ─────────────────────────────────────────────
# 17. SAVE FINAL OUTPUTS
# ─────────────────────────────────────────────
customer_df.to_csv("outputs/customer_segments.csv", index=False)
print("\n✔ Final customer_segments.csv saved.")

# Save model metadata for Flask
model_meta = {
    'best_k'         : int(best_k),
    'silhouette_score': round(final_score, 4),
    'n_customers'    : int(len(customer_df)),
    'feature_cols'   : feature_cols,
    'cluster_labels' : cluster_labels,
    'pca_variance'   : [round(float(v*100), 2) for v in explained],
}
joblib.dump(model_meta, "models/model_meta.pkl")

print("\n" + "=" * 55)
print("  PIPELINE COMPLETE ✔")
print(f"  Customers segmented : {len(customer_df)}")
print(f"  Segments found      : {best_k}")
print(f"  Silhouette Score    : {final_score:.4f}")
print("=" * 55)
