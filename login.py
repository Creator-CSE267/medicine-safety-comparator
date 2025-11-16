import streamlit as st
from user_database import update_password, verify_password, get_user
import hashlib


def password_reset(username):
    st.markdown("""
        <style>
            .reset-container {
                max-width: 420px;
                margin: auto;
                margin-top: 90px;
                padding: 35px;
                background: white;
                border-radius: 18px;
                box-shadow: 0px 4px 25px rgba(0,0,0,0.15);
                text-align: center;
            }
            .reset-title {
                font-size: 26px;
                font-weight: 700;
                color: #2E86C1;
                margin-bottom: 20px;
            }
            .reset-input input {
                border-radius: 10px !important;
                border: 2px solid #2E86C1 !important;
            }
            .reset-btn {
                width: 100%;
                border-radius: 10px;
                padding: 10px;
                font-size: 16px;
                font-weight: 600;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='reset-container'>", unsafe_allow_html=True)

    # LOGO
    st.image("logo.png", width=120)

    st.markdown("<div class='reset-title'>Change Password</div>", unsafe_allow_html=True)

    # Inputs
    old_pass = st.text_input("Current Password", type="password", placeholder="Enter current password")
    new_pass = st.text_input("New Password", type="password", placeholder="Enter new password")
    confirm_pass = st.text_input("Confirm New Password", type="password", placeholder="Re-enter new password")

    # Buttons
    col1, col2 = st.columns(2)
    update_btn = col1.button("Update Password")
    back_btn = col2.button("Back")

    st.markdown("</div>", unsafe_allow_html=True)

    # Back to login/main menu
    if back_btn:
        st.session_state["go_reset_password"] = False
        st.rerun()

    # Update logic
    if update_btn:
        if not old_pass or not new_pass or not confirm_pass:
            st.error("Please fill all fields.")
            return

        if new_pass != confirm_pass:
            st.error("New passwords do not match.")
            return

        row = get_user(username)
        if row is None:
            st.error("User not found.")
            return

        stored_hash = row[2]

        if not verify_password(old_pass, stored_hash):
            st.error("Incorrect current password.")
            return

        # Hash new password
        new_hash = hashlib.sha256(new_pass.encode()).hexdigest()

        # Update DB
        update_password(username, new_hash)

        st.success("Password updated successfully!")

        # Logout effect
        st.session_state.clear()
        st.info("Please login again with your new password.")
        st.rerun()
