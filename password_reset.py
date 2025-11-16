# password_reset.py
import streamlit as st
from user_database import get_user, update_password, verify_password

def password_reset(username: str):
    """
    Allows a logged-in user to change their password.
    Requires current password for verification.
    """
    st.header("🔑 Change / Reset Password")

    if not username:
        st.error("No user logged in.")
        return

    st.write(f"User: **{username}**")

    curr = st.text_input("Current password", type="password")
    new = st.text_input("New password", type="password")
    confirm = st.text_input("Confirm new password", type="password")

    if st.button("Update Password"):
        if not curr or not new or not confirm:
            st.warning("Please fill all fields.")
            return
        if new != confirm:
            st.error("New passwords do not match.")
            return

        row = get_user(username)
        if row is None:
            st.error("User not found.")
            return

        stored_hash = row[2]
        if not verify_password(curr, stored_hash):
            st.error("Current password is incorrect.")
            return

        ok = update_password(username, new)
        if ok:
            st.success("✅ Password updated successfully. Please log out and login again.")
        else:
            st.error("⚠️ Could not update password. Try again.")
