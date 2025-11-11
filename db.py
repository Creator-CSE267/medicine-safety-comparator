# db.py
from sqlalchemy import create_engine, Column, Integer, String, Date, DateTime, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
import pandas as pd
from datetime import datetime

# DB path (file-based SQLite)
DB_URL = "sqlite:///medsafe.db"

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# ---------- Models ----------
class Medicine(Base):
    __tablename__ = "medicines"
    id = Column(Integer, primary_key=True, index=True)
    UPC = Column(String, index=True)              # UPC as string
    Ingredient = Column(String, index=True)
    Manufacturer = Column(String)
    Batch = Column(String, index=True)
    Stock = Column(Integer, default=0)
    Expiry = Column(Date, nullable=True)
    Days_Until_Expiry = Column(Integer, nullable=True)

class Consumable(Base):
    __tablename__ = "consumables"
    id = Column(Integer, primary_key=True, index=True)
    Item_Name = Column(String, index=True)
    Category = Column(String)
    Material_Type = Column(String)
    Sterility_Level = Column(String)
    Expiry_Period_Months = Column(Integer)
    Storage_Temperature_C = Column(Float)
    Quantity_in_Stock = Column(Integer, default=0)
    Usage_Type = Column(String)
    Certification_Standard = Column(String)
    UPC = Column(String, index=True)
    Safe_Not_Safe = Column(String, default="Safe")

class UsageLog(Base):
    __tablename__ = "usage_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, server_default=func.now())
    UPC = Column(String, nullable=True)
    Ingredient = Column(String, nullable=True)
    Competitor = Column(String, nullable=True)
    Result = Column(String, nullable=True)

# ---------- Init ----------
def init_db():
    Base.metadata.create_all(bind=engine)

# ---------- CRUD Helpers ----------
def get_session():
    return SessionLocal()

# Medicines helpers
def fetch_all_medicines():
    session = get_session()
    rows = session.query(Medicine).all()
    df = pd.DataFrame([{
        "UPC": r.UPC, "Ingredient": r.Ingredient, "Manufacturer": r.Manufacturer,
        "Batch": r.Batch, "Stock": r.Stock, "Expiry": r.Expiry
    } for r in rows])
    session.close()
    return df

def upsert_medicine(upc, ingredient, manufacturer, batch, stock, expiry):
    session = get_session()
    m = session.query(Medicine).filter(Medicine.UPC == upc, Medicine.Batch == batch).first()
    if m is None:
        m = Medicine(UPC=upc, Ingredient=ingredient, Manufacturer=manufacturer,
                     Batch=batch, Stock=stock, Expiry=expiry)
        session.add(m)
    else:
        # update
        m.Stock = (m.Stock or 0) + int(stock)
        if expiry:
            m.Expiry = expiry
    session.commit()
    session.close()

def find_medicine_by_upc(upc):
    session = get_session()
    r = session.query(Medicine).filter(Medicine.UPC == upc).first()
    session.close()
    return r

def find_medicine_by_ingredient(ingredient):
    session = get_session()
    r = session.query(Medicine).filter(Medicine.Ingredient.ilike(f"%{ingredient}%")).first()
    session.close()
    return r

# Consumables helpers
def fetch_all_consumables():
    session = get_session()
    rows = session.query(Consumable).all()
    df = pd.DataFrame([{
        "Item Name": r.Item_Name, "Category": r.Category, "Material Type": r.Material_Type,
        "Sterility Level": r.Sterility_Level, "Expiry Period (Months)": r.Expiry_Period_Months,
        "Storage Temperature (C)": r.Storage_Temperature_C, "Quantity in Stock": r.Quantity_in_Stock,
        "Usage Type": r.Usage_Type, "Certification Standard": r.Certification_Standard,
        "UPC": r.UPC, "Safe/Not Safe": r.Safe_Not_Safe
    } for r in rows])
    session.close()
    return df

def upsert_consumable(item_name, category, material_type, sterility,
                      expiry_period_months, storage_temp, quantity, usage_type, cert, upc, safe_status):
    session = get_session()
    c = session.query(Consumable).filter(Consumable.UPC == upc).first()
    if c is None:
        c = Consumable(Item_Name=item_name, Category=category, Material_Type=material_type,
                       Sterility_Level=sterility, Expiry_Period_Months=expiry_period_months,
                       Storage_Temperature_C=storage_temp, Quantity_in_Stock=quantity,
                       Usage_Type=usage_type, Certification_Standard=cert, UPC=upc, Safe_Not_Safe=safe_status)
        session.add(c)
    else:
        c.Quantity_in_Stock = (c.Quantity_in_Stock or 0) + int(quantity)
        c.Expiry_Period_Months = expiry_period_months
    session.commit()
    session.close()

# Logs
def append_log(upc, ingredient, competitor, result):
    session = get_session()
    log = UsageLog(UPC=upc, Ingredient=ingredient, Competitor=competitor, Result=result)
    session.add(log)
    session.commit()
    session.close()

def fetch_recent_logs(limit=50):
    session = get_session()
    rows = session.query(UsageLog).order_by(UsageLog.id.desc()).limit(limit).all()
    df = pd.DataFrame([{
        "timestamp": r.timestamp, "UPC": r.UPC, "Ingredient": r.Ingredient,
        "Competitor": r.Competitor, "Result": r.Result
    } for r in rows])
    session.close()
    return df

# Utility: import CSVs (one-off migration helper)
def import_csv_to_db(medicine_csv=None, inventory_csv=None, consumables_csv=None):
    session = get_session()
    if medicine_csv:
        dfm = pd.read_csv(medicine_csv, dtype={"UPC": str})
        for _, r in dfm.iterrows():
            try:
                expiry = None
                if "Expiry" in r and pd.notna(r["Expiry"]):
                    expiry = pd.to_datetime(r["Expiry"]).date()
                # create medicine rows (leave Batch blank if not present)
                m = Medicine(UPC=str(r.get("UPC", "")), Ingredient=r.get("Active Ingredient", r.get("Ingredient", "")),
                             Manufacturer=r.get("Manufacturer", None), Batch=r.get("Batch", None),
                             Stock=int(r.get("Quantity", r.get("Stock", 0)) or 0), Expiry=expiry)
                session.add(m)
            except Exception:
                continue
    if inventory_csv:
        di = pd.read_csv(inventory_csv, dtype={"UPC": str})
        for _, r in di.iterrows():
            try:
                expiry = None
                if "Expiry" in r and pd.notna(r["Expiry"]):
                    expiry = pd.to_datetime(r["Expiry"]).date()
                m = Medicine(UPC=str(r.get("UPC", "")), Ingredient=r.get("Ingredient", ""),
                             Manufacturer=r.get("Manufacturer", None), Batch=r.get("Batch", None),
                             Stock=int(r.get("Stock", 0) or 0), Expiry=expiry)
                session.add(m)
            except Exception:
                continue
    if consumables_csv:
        dc = pd.read_csv(consumables_csv, dtype={"UPC": str})
        for _, r in dc.iterrows():
            try:
                c = Consumable(Item_Name=r.get("Item Name", ""),
                               Category=r.get("Category", ""),
                               Material_Type=r.get("Material Type", ""),
                               Sterility_Level=r.get("Sterility Level", ""),
                               Expiry_Period_Months=int(r.get("Expiry Period (Months)", 0) or 0),
                               Storage_Temperature_C=float(r.get("Storage Temperature (C)", 0) or 0),
                               Quantity_in_Stock=int(r.get("Quantity in Stock", 0) or 0),
                               Usage_Type=r.get("Usage Type", ""),
                               Certification_Standard=r.get("Certification Standard", ""),
                               UPC=str(r.get("UPC", "")),
                               Safe_Not_Safe=r.get("Safe/Not Safe", "Safe"))
                session.add(c)
            except Exception:
                continue
    session.commit()
    session.close()
