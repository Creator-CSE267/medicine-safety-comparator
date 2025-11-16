# login.py
import streamlit as st
from user_database import get_user, verify_password
from password_reset import password_reset
from datetime import datetime

def login_router():
    """Renders login or reset password UI. When login succeeds, sets session_state and reruns."""
    if "do_reset" not in st.session_state:
        st.session_state["do_reset"] = False

    if st.session_state["do_reset"]:
        user = st.session_state.get("reset_username", "")
        password_reset(user)
        return

    return login_page()


def login_page():
    st.markdown("""
    <style>
        body { background: linear-gradient(135deg, #2E86C1 0%, #5DADE2 100%); }
        .login-card {
            max-width: 480px;
            margin: 70px auto;
            padding: 36px 42px;
            background: rgba(255,255,255,0.12);
            border-radius: 18px;
            backdrop-filter: blur(8px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            animation: floatIn 0.7s ease-out;
        }
        .login-title { text-align:center; font-size:26px; font-weight:800; color:#fff; margin-bottom:8px;}
        .login-sub { text-align:center; color:#EAF6FF; margin-bottom:18px; }
        .login-input input { border-radius:10px !important; border:2px solid rgba(255,255,255,0.25) !important; background: rgba(255,255,255,0.95) !important; color:#123; }
        .btn-row { display:flex; gap:8px; }
        @keyframes floatIn { from { transform: translateY(18px); opacity:0 } to { transform: translateY(0); opacity:1 } }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='login-card'>", unsafe_allow_html=True)

    # logo (will show if logo.png exists)
    try:
        st.image("logo.png", width=110)
    except Exception:
        pass

    st.markdown("<div class='login-title'>Welcome to MedSafe AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='login-sub'>Secure access for pharmacists & admins</div>", unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter your username", key="login_username")
    password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")

    login_btn = st.button("Login", key="login_btn")
    col1, col2 = st.columns([1,1])
    with col1:
        reset_form_btn = st.button("Reset", key="reset_form_btn")
    with col2:
        change_pass_btn = st.button("Change Password", key="change_pass_btn")

    st.markdown("</div>", unsafe_allow_html=True)

    if reset_form_btn:
        # clear fields
        st.session_state["login_username"] = ""
        st.session_state["login_password"] = ""
        st.experimental_rerun()

    if change_pass_btn:
        if not username:
            st.error("Enter username first to reset password.")
            return
        st.session_state["do_reset"] = True
        st.session_state["reset_username"] = username
        st.experimental_rerun()

    if login_btn:
        if not username or not password:
            st.error("Enter both username and password.")
            return

        row = get_user(username)
        if row is None:
            st.error("User not found.")
            return

        stored_hash = row[2]
        role = row[3]

        if verify_password(password, stored_hash):
            # SUCCESS - set session and rerun
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["role"] = role
            # set last_active for timeout
            st.session_state["last_active"] = datetime.now().isoformat()
            st.success("Login successful — redirecting...")
            st.experimental_rerun()
        else:
            st.error("Incorrect password.")
