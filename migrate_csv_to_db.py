# migrate_csv_to_db.py
from db import init_db, import_csv_to_db
import os

init_db()
# Update paths if your CSVs are in a folder
medicine_csv = "medicine_dataset.csv" if os.path.exists("medicine_dataset.csv") else None
inventory_csv = "inventory.csv" if os.path.exists("inventory.csv") else None
consumables_csv = "consumables_dataset.csv" if os.path.exists("consumables_dataset.csv") else None

import_csv_to_db(medicine_csv=medicine_csv, inventory_csv=inventory_csv, consumables_csv=consumables_csv)
print("Migration finished. DB file created: medsafe.db")
