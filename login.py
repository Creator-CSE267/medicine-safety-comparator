import streamlit as st
from user_database import get_user, verify_password
from password_reset import password_reset


# ---------------------------------------------------
#  PROFESSIONAL LOGIN PAGE
# ---------------------------------------------------
def login_page():
    st.markdown("""
        <style>
            .login-container {
                width: 420px;
                margin: auto;
                margin-top: 100px;
                padding: 35px;
                background: white;
                border-radius: 18px;
                box-shadow: 0px 4px 25px rgba(0,0,0,0.20);
                text-align: center;
            }
            .login-title {
                font-size: 28px;
                font-weight: 800;
                color: #2E86C1;
                margin-bottom: 20px;
            }
            .login-input input {
                border-radius: 10px !important;
                border: 2px solid #2E86C1 !important;
            }
            .login-btn {
                width: 100%;
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
                font-weight: 600;
            }
        </style>
    """, unsafe_allow_html=True)

    # -------- LOGIN BOX --------
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)

    st.image("logo.png", width=130)

    st.markdown("<div class='login-title'>MedSafe Login</div>", unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter password")

    login_btn = st.button("Login", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    reset_btn = st.button("Reset Password", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------- RESET PASSWORD --------
    if reset_btn:
        st.session_state["go_reset_password"] = True
        st.session_state["reset_user"] = username
        st.rerun()

    # -------- LOGIN ACTION --------
    if login_btn:
        if username.strip() == "" or password.strip() == "":
            st.error("Please enter username and password.")
            return None, None

        row = get_user(username)

        if row is None:
            st.error("User not found.")
            return None, None

        stored_hash = row[2]
        role = row[3]

        if verify_password(password, stored_hash):
            # Login Success
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["role"] = role
            st.rerun()
        else:
            st.error("Incorrect password.")
            return None, None

    return None, None



# ---------------------------------------------------
#  ROUTER FUNCTION
# ---------------------------------------------------
def login_router():
    """Handles switching between login → reset password"""
    if "go_reset_password" not in st.session_state:
        st.session_state["go_reset_password"] = False

    if st.session_state["go_reset_password"]:
        user = st.session_state.get("reset_user", "")
        password_reset(user)
        return None, None

    return login_page()
