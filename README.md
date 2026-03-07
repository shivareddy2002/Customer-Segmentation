 <p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:8e2de2,100:4a00e0&height=180&section=header&text=📝%20Amazon%20Customer%20Using%2Segmentation&fontSize=34&fontColor=ffffff&animation=fadeIn&fontAlignY=35"/>
</p> 
## Hackathon ML Project — End-to-End Production Pipeline

This project builds an **end-to-end machine learning pipeline** to segment Amazon customers based on purchasing behavior. The system uses **unsupervised learning algorithms** to discover hidden customer groups and helps businesses perform **targeted marketing and personalized recommendations**.

## Project DEMO LINK: https://customer-segmentation-three.vercel.app/

The project also includes a **Flask dashboard** that visualizes customer segments and allows **real-time prediction for new customers**.

---

# 📌 Problem Statement

Large e-commerce platforms like Amazon have millions of customers with different purchasing behaviors.

Without segmentation it becomes difficult to:

- Target the right customers
- Personalize recommendations
- Improve marketing ROI
- Identify high-value customers

This project solves the problem by applying **machine learning clustering techniques to identify meaningful customer segments**.

---

# 🎯 Project Objective

The main goal of this project is to:

- Identify distinct **customer segments**
- Understand **customer purchasing behavior**
- Enable **targeted marketing strategies**
- Provide **data-driven insights for business decisions**

---

# 🗂️ Project Structure

## 🗂️ Project Structure
```
amazon_customer_segmentation/
│
├── data/
│   └── amazon_sales_clustering_dataset.csv
│
├── models/                    ← Saved ML artifacts
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── pca.pkl
│   ├── feature_cols.pkl
│   ├── cluster_labels.pkl
│   └── model_meta.pkl
│
├── outputs/
│   ├── customer_segments.csv
│   ├── cluster_summary.csv
│   └── category_recommendations.csv
│
├── static/           
│   ├── css
│   ├── js
│   ├── plots ← 10 generated charts
├── templates/                 ← Flask HTML pages
│   ├── base.html
│   ├── home.html
│   ├── about.html
│   ├── workflow.html
│   ├── dashboard.html
│   └── predict.html
│
├── clustering_pipeline.py     ← Full ML pipeline
├── app.py                     ← Flask dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python clustering_pipeline.py
```

### 3. Start the Flask dashboard
```bash
python app.py
```

### 4. Open browser
```
http://127.0.0.1:5000
```

---

## 🔬 ML Pipeline Steps

| Step | Task |
|------|------|
| 1 | Load & Clean Data |
| 2 | Feature Engineering (total_spent, date parts) |
| 3 | RFM Model (Recency, Frequency, Monetary) |
| 4 | Behavioral Features (14 total) |
| 5 | StandardScaler normalization |
| 6 | Elbow + Silhouette for optimal k |
| 7 | K-Means (primary) |
| 8 | Hierarchical Clustering (dendrogram) |
| 9 | DBSCAN (noise detection) |
| 10 | PCA 2D visualization |
| 11 | 10 analytical plots |
| 12 | Cluster labeling & recommendation engine |
| 13 | Save models + Flask deployment |

---

## 📊 Customer Segments

| Segment | Strategy |
|---------|----------|
| 💎 High-Value Customers | VIP program, exclusive access |
| 🔁 Frequent Buyers | Subscriptions, bundle deals |
| 🏷️ Discount Shoppers | Coupons, flash sales |
| 📅 Occasional Buyers | Retargeting, win-back campaigns |

---


---

## 🌐 Flask Web Application

The project includes a **Flask-based interactive dashboard.**

### Available Pages

| Route | Description |
|------|-------------|
| `/` | Landing page with project overview |
| `/about` | Problem statement & ML algorithms |
| `/workflow` | End-to-end ML pipeline explanation |
| `/dashboard` | Customer segmentation analytics |
| `/predict` | Real-time prediction for new customers |

---

## 🧪 Tech Stack

### Programming
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- SciPy

### Visualization
- Matplotlib
- Seaborn

### Web Application
- Flask
- HTML
- CSS
- JavaScript

### Model Persistence
- Joblib

---

## 💡 Business Impact

This system helps businesses:

- Identify **high-value customers**
- Improve **targeted marketing campaigns**
- Increase **customer retention**
- Enhance **personalized recommendations**
- Make **data-driven marketing decisions**

---

## 🚀 Future Improvements

Potential improvements for this project:

- Deep learning based segmentation
- Real-time data pipeline
- Cloud deployment (AWS / GCP)
- Interactive BI dashboards
- Recommendation system integration

---
## 👨‍💻 Author  

**Lomada Siva Gangi Reddy**  
- 🎓 B.Tech CSE (Data Science), RGMCET (2021–2025)  
- 💡 Interests: Python | Machine Learning | Deep Learning | Data Science  
- 📍 Open to **Internships & Job Offers**

 **Contact Me**:  

- 📧 **Email**: lomadasivagangireddy3@gmail.com  
- 📞 **Phone**: 9346493592  
- 💼 [LinkedIn](https://www.linkedin.com/in/lomada-siva-gangi-reddy-a64197280/)  🌐 [GitHub](https://github.com/shivareddy2002)  🚀 [Portfolio](https://lsgr-portfolio-pulse.lovable.app/)

---
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:f9c74f,100:ff4b4b&height=120&section=footer"/>
</p>
