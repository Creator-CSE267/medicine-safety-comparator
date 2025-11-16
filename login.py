import streamlit as st
from user_database import get_user, verify_password

def login_page():
    st.markdown("""
        <style>
            .login-container {
                max-width: 420px;
                margin: auto;
                margin-top: 90px;
                padding: 35px;
                background: white;
                border-radius: 18px;
                box-shadow: 0px 4px 25px rgba(0,0,0,0.15);
                text-align: center;
            }
            .login-title {
                font-size: 28px;
                font-weight: 700;
                color: #2E86C1;
                margin-bottom: 15px;
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

    # Centered Login Box
    with st.container():

        st.markdown("<div class='login-container'>", unsafe_allow_html=True)

        # LOGO
        st.image("logo.png", width=120)

        st.markdown("<div class='login-title'>User Login</div>", unsafe_allow_html=True)

        # Username Input
        username = st.text_input("Username", key="login_username", placeholder="Enter username")

        # Password Input
        password = st.text_input("Password", type="password", key="login_password", placeholder="Enter password")

        # Buttons layout
        col1, col2, col3 = st.columns(3)

        login_btn = col1.button("Login")
        reset_btn = col2.button("Reset")
        change_btn = col3.button("Change Password")

        st.markdown("</div>", unsafe_allow_html=True)

    # RESET clears fields
    if reset_btn:
        st.session_state["login_username"] = ""
        st.session_state["login_password"] = ""
        st.rerun()

    # Change Password Flow
    if change_btn:
        st.session_state["go_reset_password"] = True
        st.rerun()

    # HANDLE LOGIN
    if login_btn:
        if not username or not password:
            st.error("Please enter both username and password.")
            return None, None

        row = get_user(username)

        if row is None:
            st.error("Invalid username.")
            return None, None

        stored_hash = row[2]

        if verify_password(password, stored_hash):
            st.success("Login successful.")

            # Save session state
            st.session_state["logged_in"] = True
            st.session_state["username"] = row[0]
            st.session_state["role"] = row[3]

            st.rerun()
        else:
            st.error("Incorrect password.")

    # If not logged in
    if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
        return None, None

    # Return successful login
    return st.session_state["username"], st.session_state["role"]
