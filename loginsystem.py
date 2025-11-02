
from dotenv import load_dotenv
from supabase import create_client
import os
load_dotenv()
from datetime import datetime

import streamlit as st
SECRETKEY = os.getenv("API_KEY")
DATABASE_URL = os.getenv("PROJECT_URL")
supabase = create_client(DATABASE_URL, SECRETKEY)
USERS = {
    "user1": "pass123",
    "user2": "abc456"
}


# ---------------------------
# ✅ Sign Up Page
# ---------------------------
def signup_page():
    st.subheader("Create a New Account")
    new_email = st.text_input("Email")
    new_password = st.text_input("Password", type="password")

    if st.button("Sign Up"):
        try:
            response = supabase.auth.sign_up({
                "email": new_email,
                "password": new_password
            })

            if response.user:
                st.success("✅ Account created successfully! Please check your email for verification.")

                # Optional: Add user record to database
                supabase.table("users").insert({
                    "email": new_email,
                    "created_at": datetime.utcnow().isoformat()
                }).execute()

            else:
                st.error("Signup failed. Please try again.")

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------
# ✅ Login Page
# ---------------------------
def login_page():
    st.subheader("Login to your Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            if response.user:
                st.session_state["logged_in"] = True
                st.session_state["username"] = response.user.email
                st.success(f"Welcome, {response.user.email}!")
                st.switch_page("pages/main.py")  # redirect to your main app page
            else:
                st.error("Invalid credentials.")
        except Exception as e:
            st.error(f"Error: {e}")


# ---------------------------
# ✅ Protected Content
# ---------------------------
def protected_content():
    st.title("🔒 Protected Area")
    st.write(f"Welcome, {st.session_state.username}!")

    if st.button("Logout"):
        try:
            supabase.auth.sign_out()
        except Exception as e:
            st.warning(f"Error signing out: {e}")
        st.session_state.clear()
        st.experimental_rerun()

# ---------------------------
# ✅ Sidebar Navigation
# ---------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

st.sidebar.title("🔐 Navigation")

if st.session_state["logged_in"]:
    protected_content()
else:
    choice = st.sidebar.radio("Go to", ["Login", "Signup"])

    if choice == "Login":
        login_page()
    elif choice == "Signup":
        signup_page()