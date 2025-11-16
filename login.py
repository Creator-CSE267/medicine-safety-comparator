# login.py
import streamlit as st
from user_database import get_user
from passlib.context import CryptContext
from streamlit_cookies_manager import EncryptedCookieManager

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def login_page():
    st.title("🔐 Login")

    cookies = EncryptedCookieManager(
        prefix="medsafe_login_",
        password="your_cookie_key_123"
    )
    if not cookies.ready():
        st.stop()

    # If session exists → auto-login
    if "logged_in" in st.session_state and st.session_state["logged_in"]:
        return st.session_state["username"], st.session_state["role"]

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = get_user(username)

        if user and pwd_context.verify(password, user[2]):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = user[3]
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    st.stop()
