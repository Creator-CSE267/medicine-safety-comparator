# password_reset.py
import streamlit as st
from user_database import update_password
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def password_reset(username):
    st.header("🔑 Change Your Password")

    old_pw = st.text_input("Current Password", type="password")
    new_pw = st.text_input("New Password", type="password")
    confirm_pw = st.text_input("Confirm New Password", type="password")

    if st.button("Update Password"):
        if new_pw != confirm_pw:
            st.error("❌ New passwords do not match")
            return

        new_hash = pwd_context.hash(new_pw)
        update_password(username, new_hash)
        st.success("✅ Password updated successfully")
