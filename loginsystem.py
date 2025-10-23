# import streamlit as st

# # In a real application, you would store users in a database with hashed passwords
# USERS = {
#     "user1": "pass123",
#     "user2": "abc456"
# }

# def signup_page():
#     st.subheader("Create a New Account")
#     new_username = st.text_input("New Username")
#     new_password = st.text_input("New Password", type="password")
#     if st.button("Sign Up"):
#         if new_username in USERS:
#             st.error("Username already exists.")
#         else:
#             USERS[new_username] = new_password # In real app, hash password
#             st.success("Account created successfully! Please log in.")

# def login_page():
#     st.subheader("Login to your Account")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     if st.button("Login"):
#         if username in USERS and USERS[username] == password:
#             st.session_state["logged_in"] = True
#             st.session_state["username"] = username
#             st.success(f"Welcome, {username}!")
#             st.rerun() # Refresh to show protected content
#         else:
#             st.error("Invalid username or password.")

# def protected_content():
#     st.title("Welcome to the Protected Area!")
#     st.write(f"Hello, {st.session_state.username}!")
#     if st.button("Logout"):
#         del st.session_state["logged_in"]
#         del st.session_state["username"]
#         st.experimental_rerun()

# if "logged_in" not in st.session_state:
#     st.session_state["logged_in"] = False

# if st.session_state["logged_in"]:
#     st.switch_page("pages/app.py") # Redirect to main app page
# else:
#     st.sidebar.title("Navigation")
#     choice = st.sidebar.radio("Go to", ["Login", "Signup"])
#     if choice == "Login":
#         login_page()
#     elif choice == "Signup":
#         signup_page()

# # if "logged_in" not in st.session_state:
# #     st.session_state.logged_in = False

# # if not st.session_state.logged_in:
# #     login_page()
# # else:
# #     st.switch_page("app.py") # Redirect if already logged in


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
# ✅ Password Reset Page
# ---------------------------
def password_reset_page():
    st.subheader("Reset Your Password")
    email = st.text_input("Enter your email")

    if st.button("Send Reset Link"):
        try:
            supabase.auth.reset_password_email(email)
            st.success("📩 Password reset email sent! Check your inbox.")
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
    choice = st.sidebar.radio("Go to", ["Login", "Signup", "Forgot Password"])

    if choice == "Login":
        login_page()
    elif choice == "Signup":
        signup_page()
    elif choice == "Forgot Password":
        password_reset_page()
