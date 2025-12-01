
# **MedSafe AI – Medicine Safety Comparator**

A professional Streamlit-based application that evaluates the safety of competitor medicines by comparing them against standard reference medicines using laboratory parameters, deterministic safety rules, and an optional machine-learning model.  
It also includes a complete dashboard, inventory management, secure login system, PDF reporting, and MongoDB-backed logging.

---

## 🚀 **Key Features**

### 🧪 **Smart Medicine Testing**
- Compare competitor samples against standard medicines.
- Strict rule-based evaluation (must pass *all* criteria to be marked Safe).
- Handles **higher-is-better** and **lower-is-better** metrics correctly.
- Special “optimal zone” handling for **Assay Purity (%)** (both minimum & maximum limits matter).
- Per-criterion PASS/FAIL details.
- Automatic “Suggested Improvements” based on failed parameters.

### 🤖 **Optional Machine Learning (RandomForest)**
- Trains using data from the `medicines` MongoDB collection.
- Feature engineering based on absolute and delta values.
- TF-IDF text processing for ingredient/use-case.
- Automatically falls back to a rule-based comparator if model training fails.

### 📊 **Professional Dashboard**
- KPI indicators (Total Tests, Safe, Not Safe, and more).
- Daily trend charts.
- Per-ingredient analysis.
- Paginated result tables with search & filtering.
- Download logs as CSV.
- Admin-only actions (Clear logs).

### 📦 **Inventory Management**
- Medicines & Consumables tabs.
- Add / edit / delete functions.
- Search & filters (UPC, ingredient, batch, low stock/low quantity).
- Expiry tracking (Expired / Expiring soon / Valid).

### 📄 **PDF Report Generation**
- Includes logos, prediction summary, competitor vs standard table, suggestions, and chart.

### 🔐 **Authentication System**
- Login system with user roles (Admin, Pharmacist).
- Password reset module.
- Session timeout management.

---

## 📁 **Project Structure**

```
medicine-safety-comparator/
│── app.py
│── styles.py
│── login.py
│── user_database.py
│── password_reset.py
│── models/
│── avatars/
│── assets/
│── requirements.txt
│── README.md
```

---

## 🛠 **Setup Instructions**

### 1. Clone
```bash
git clone https://github.com/Creator-CSE267/medicine-safety-comparator.git
cd medicine-safety-comparator
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Configure MongoDB
```
[MONGO]
URI = "your-mongo-uri"
DBNAME = "your-db-name"
```

### 4. Run
```bash
streamlit run app.py
```

---

## 🧬 **Safety Evaluation Logic**

### ✔ Higher-is-better  
- Days Until Expiry  
- Dissolution Rate (%)  
- Warning Labels  

### ✔ Lower-is-better  
- Disintegration Time  
- Impurity Level  

### ✔ Assay Purity (%)  
Must be within 90–105%.

### ✔ Strict Rule  
A medicine is **Safe only if all metrics pass**.

---

## 📊 Dashboard Features

- KPIs  
- Date filters  
- Per-ingredient charts  
- Trend line  
- Search  
- CSV downloads  
- Admin log-clear button  

---

## 📦 Inventory Features

### Medicines
- Track stock, expiry, batch, manufacturer
- Filters + low-stock alerts

### Consumables
- Track category, quantity, safety status
- Filters + low-quantity alerts

---

## 🐞 Troubleshooting

### Invalid date comparison
Ensure expiry values are ISO dates.

### StreamlitDuplicateElementId
Assign unique `key=` to widgets inside loops.

---

## 📜 License
MIT License  

---

## 📧 Contact
Author: **Creator-CSE267**  
GitHub: https://github.com/Creator-CSE267
