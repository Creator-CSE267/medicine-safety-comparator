MedSafe AI – Medicine Safety Comparator

A professional Streamlit-based application that evaluates the safety of competitor medicines by comparing them against standard reference medicines using laboratory parameters, deterministic safety rules, and an optional machine-learning model.
It also includes a complete dashboard, inventory management, secure login system, PDF reporting, and MongoDB-backed logging.

🚀 Key Features
🧪 Smart Medicine Testing

Compare competitor samples against standard medicines.

Strict rule-based evaluation (must pass all criteria to be marked Safe).

Handles higher-is-better and lower-is-better metrics correctly.

Special “optimal zone” handling for Assay Purity (%) (both minimum & maximum limits matter).

Per-criterion PASS/FAIL details.

Automatic “Suggested Improvements” based on failed parameters.

🤖 Optional Machine Learning (RandomForest)

Trains using data from the medicines MongoDB collection.

Feature engineering based on absolute and delta values.

TF-IDF text processing for ingredient/use-case.

Automatically falls back to a rule-based comparator if model training fails.

📊 Professional Dashboard

KPI indicators (Total Tests, Safe, Not Safe, and more).

Daily trend charts.

Per-ingredient analysis.

Paginated result tables with search & filtering.

Download logs as CSV.

Admin-only actions (Clear logs).

📦 Inventory Management

Medicines & Consumables tabs.

Add / edit / delete functions.

Search & filters (UPC, ingredient, batch, low stock/low quantity).

Expiry tracking (Expired / Expiring soon / Valid).

📄 PDF Report Generation

Includes logos, prediction summary, competitor vs standard table, suggestions, and chart.

🔐 Authentication System

Login system with user roles (Admin, Pharmacist).

Password reset module.

Session timeout management.

📁 Project Structure
medicine-safety-comparator/
│── app.py                # Main Streamlit application
│── styles.py             # UI themes & global CSS
│── login.py              # Login handling
│── user_database.py      # User credentials DB functions
│── password_reset.py     # Password reset flow
│── models/               # Saved ML models
│── avatars/              # User profile images
│── assets/               # Logo & background images
│── requirements.txt      # Dependencies
│── README.md             # Documentation

🛠 Setup Instructions
1. Clone the repository
git clone https://github.com/Creator-CSE267/medicine-safety-comparator.git
cd medicine-safety-comparator

2. Install dependencies
pip install -r requirements.txt

3. Configure MongoDB

Add these to your environment or Streamlit secrets:

[MONGO]
URI = "your-mongo-uri"
DBNAME = "your-db-name"

4. Run the application
streamlit run app.py

🧬 How the Safety Evaluation Works

Each competitor medicine is compared parameter-by-parameter against the standard.

✔ Higher-is-better metrics

Days Until Expiry

Dissolution Rate (%)

Warning Labels Present

Competitor value must be ≥ standard value.

✔ Lower-is-better metrics

Disintegration Time (minutes)

Impurity Level (%)

Competitor value must be ≤ standard value.

✔ Special case: Assay Purity (%)

Must fall within an acceptable min–max range
Example: 90% – 105%

Too high is also bad (possible over-concentration / analytical error).

✔ Strict rule

A medicine is marked Safe only if all criteria pass.
Any single failure → Not Safe.

The system also generates professional suggestions for every failed criterion.

📊 Dashboard Highlights

KPIs: Total tests, Safe vs Not Safe, Expired vs Expiring Soon, etc.

Filters:

Date range

Ingredient

Result type

Free text search (UPC / Competitor)

Charts:

Daily test trends

Tests per ingredient

Data Table:
Paginated, searchable, sortable results.

Export Options:
Download filtered or full logs as CSV.

Admin:
Clear logs with one click.

📦 Inventory Module
Medicines

Track stock, expiry, manufacturer, batch, UPC.

Alerts for expired & expiring soon.

Filters: UPC, ingredient, manufacturer, batch, low stock.

Consumables

Track item name, category, quantity, storage temp.

Filters: name, category, UPC, safe/not safe, low quantity.

🐞 Troubleshooting
1. TypeError: comparison between dtype=date and dtype=str

Fix: ensure Expiry dates are stored in ISO format (YYYY-MM-DD) and parsed with:

pd.to_datetime(..., errors="coerce").dt.date

2. StreamlitDuplicateElementId

Cause: duplicate widget without unique key.
Fix: add key="unique_key_here" to widgets inside loops/expanders.

3. Model training failure

Cause: dataset too small or missing classes.
Fix: add at least one Safe + one Not Safe sample per ingredient.

📘 PDF Report Contents

Logo

Standard medicine info

Competitor details

Prediction result

Suggestions

Per-metric comparison

Chart

📜 License

MIT License (or update to your preferred license).

📧 Contact

Author: Creator-CSE267
GitHub: https://github.com/Creator-CSE267
