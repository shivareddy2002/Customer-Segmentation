"""
============================================================
  Amazon Customer Segmentation - Flask Dashboard
  Routes: Home | About | Workflow | Dashboard | Predict
============================================================
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ── Load models & data once at startup ──────────────────────
scaler       = joblib.load("models/scaler.pkl")
kmeans       = joblib.load("models/kmeans_model.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
cluster_lbls = joblib.load("models/cluster_labels.pkl")
model_meta   = joblib.load("models/model_meta.pkl")
pca_model    = joblib.load("models/pca.pkl")

customer_df  = pd.read_csv("outputs/customer_segments.csv")
cluster_sum  = pd.read_csv("outputs/cluster_summary.csv")
cat_reco     = pd.read_csv("outputs/category_recommendations.csv")

SEGMENT_INFO = {
    "High-Value Customers": {
        "icon"    : "💎",
        "color"   : "#E94560",
        "strategy": "VIP loyalty program, exclusive early access, personalized concierge",
        "desc"    : "Top spenders with high purchase frequency. Most profitable segment.",
    },
    "Frequent Buyers": {
        "icon"    : "🔁",
        "color"   : "#236CC6",
        "strategy": "Subscription plans, bundle offers, tier-based reward points",
        "desc"    : "Buy often but mid-range spending. High retention potential.",
    },
    "Discount Shoppers": {
        "icon"    : "🏷️",
        "color"   : "#27946C",
        "strategy": "Flash sales, coupon campaigns, bulk-buy discounts",
        "desc"    : "Price-sensitive buyers who respond strongly to promotions.",
    },
    "Occasional Buyers": {
        "icon"    : "📅",
        "color"   : "#89B41D",
        "strategy": "Retargeting ads, seasonal campaigns, win-back email series",
        "desc"    : "Infrequent purchases. High recency — needs re-engagement.",
    },
}

# ── Helper ───────────────────────────────────────────────────
def get_segment_color(name):
    return SEGMENT_INFO.get(name, {}).get("color", "#888")

def build_dashboard_stats():
    stats = {}
    total = len(customer_df)
    for cl, name in cluster_lbls.items():
        count = int((customer_df['cluster'] == cl).sum())
        pct   = round(count / total * 100, 1)
        sub   = customer_df[customer_df['cluster'] == cl]
        stats[name] = {
            "count"    : count,
            "pct"      : pct,
            "color"    : get_segment_color(name),
            "icon"     : SEGMENT_INFO.get(name, {}).get("icon", "👤"),
            "avg_spend": round(float(sub['Monetary'].mean()), 2),
            "avg_freq" : round(float(sub['Frequency'].mean()), 2),
            "strategy" : SEGMENT_INFO.get(name, {}).get("strategy", ""),
        }
    return stats

# ── ROUTES ───────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html", meta=model_meta)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/workflow")
def workflow():
    return render_template("workflow.html")


@app.route("/dashboard")
def dashboard():
    stats     = build_dashboard_stats()
    plots     = [f for f in os.listdir("static/plots") if f.endswith(".png")]
    plots_info = [
        {"file": "frequency_dist.png",     "title": "Frequency Distribution"},
        {"file": "monetary_dist.png",      "title": "Monetary Distribution"},
        {"file": "pca_clusters.png",       "title": "PCA 2D Cluster Visualization"},
        {"file": "recency_dist.png",       "title": "Recency Distribution"},
        {"file": "cluster_count.png",      "title": "Customer Count per Segment"},
        {"file": "recency_monetary.png",   "title": "Recency vs Spending"},
        {"file": "freq_monetary.png",      "title": "Frequency vs Spending"},
        {"file": "correlation_heatmap.png","title": "Feature Correlation Heatmap"},
        {"file": "rating_cluster.png",     "title": "Rating by Segment"},
        {"file": "elbow_silhouette.png",  "title": "Elbow & Silhouette Analysis"},
    ]
    recs = cat_reco.to_dict(orient='records')
    return render_template("dashboard.html",
                           stats=stats, meta=model_meta,
                           plots=plots_info, recs=recs,
                           cluster_lbls=cluster_lbls)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        try:
            inp = {
                "Recency"          : float(request.form["recency"]),
                "Frequency"        : float(request.form["frequency"]),
                "Monetary"         : float(request.form["monetary"]),
                "total_quantity"   : float(request.form["total_quantity"]),
                "avg_price"        : float(request.form["avg_price"]),
                "avg_rating"       : float(request.form["avg_rating"]),
                "total_reviews"    : float(request.form["total_reviews"]),
                "unique_categories": float(request.form["unique_categories"]),
                "avg_order_value"  : float(request.form["avg_order_value"]),
                "max_single_spend" : float(request.form["max_single_spend"]),
                "purchase_days"    : float(request.form["purchase_days"]),
                "weekend_purchases": float(request.form["weekend_purchases"]),
                "engagement_score" : float(request.form["avg_rating"]) * np.log1p(float(request.form["total_reviews"])),
                "loyalty_index"    : float(request.form["frequency"]) / (float(request.form["recency"]) + 1),
            }

            row = pd.DataFrame([inp])[feature_cols]
            scaled_row = scaler.transform(row)
            cluster_id = int(kmeans.predict(scaled_row)[0])
            seg_name   = cluster_lbls.get(cluster_id, f"Cluster {cluster_id}")
            info       = SEGMENT_INFO.get(seg_name, {})

            # distances to all centroids
            distances = []
            for i, center in enumerate(kmeans.cluster_centers_):
                d = float(np.linalg.norm(scaled_row[0] - center))
                distances.append({"cluster": i,
                                  "name": cluster_lbls.get(i, f"Cluster {i}"),
                                  "distance": round(d, 4)})
            distances.sort(key=lambda x: x["distance"])

            result = {
                "cluster_id"  : cluster_id,
                "segment_name": seg_name,
                "icon"        : info.get("icon", "👤"),
                "color"       : info.get("color", "#888"),
                "desc"        : info.get("desc", ""),
                "strategy"    : info.get("strategy", ""),
                "distances"   : distances,
                "input"       : inp,
            }
        except Exception as e:
            result = {"error": str(e)}

    return render_template("predict.html", result=result)


@app.route("/api/stats")
def api_stats():
    stats = build_dashboard_stats()
    return jsonify(stats)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
