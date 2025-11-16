# login.py
import streamlit as st
from user_database import init_user_db, get_user, verify_password
from typing import Tuple

# Re-export init_user_db so app.py can import it from login
__all__ = ["init_user_db", "login_page"]

def _ensure_session():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "role" not in st.session_state:
        st.session_state["role"] = None


def logout():
    """Clear session login keys and rerun"""
    for k in ["logged_in", "username", "role"]:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()


def login_page() -> Tuple[str, str]:
    """
    Show login UI. If user is already logged in in session_state, return (username, role).
    Otherwise show login form and return (None, None) after stopping the app so the
    main app does not proceed until login.
    """
    _ensure_session()

    # If already logged in, show sidebar logout and return credentials
    if st.session_state.get("logged_in"):
        # show logout in sidebar
        if st.sidebar.button("Logout"):
            logout()
        return st.session_state.get("username"), st.session_state.get("role")

    st.title("🔐 MedSafe AI — Login")

    col1, col2 = st.columns([2, 1])
    with col1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
    with col2:
        if st.button("Login"):
            if not username or not password:
                st.error("Enter both username and password.")
            else:
                row = get_user(username)
                if row is None:
                    st.error("Invalid username or password.")
                else:
                    stored_hash = row[2]
                    if verify_password(password, stored_hash):
                        st.success("Login successful.")
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = row[0]
                        st.session_state["role"] = row[3]
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

    # Not logged in yet -> stop app so the rest of app doesn't run
    st.stop()
    return None, None  # unreachable, but explicit
