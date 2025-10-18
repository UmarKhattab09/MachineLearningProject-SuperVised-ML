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
    


import streamlit as st

USERS = {
    "user1": "pass123",
    "user2": "abc456"
}

def signup_page():
    st.subheader("Create a New Account")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")
    if st.button("Sign Up"):
        if new_username in USERS:
            st.error("Username already exists.")
        else:
            USERS[new_username] = new_password # In real app, hash password
            st.success("Account created successfully! Please log in.")

def login_page():
    st.subheader("Login to your Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success(f"Welcome, {username}!")
            st.switch_page("pages/main.py")  # <--- Only redirect after successful login
        else:
            st.error("Invalid username or password.")

def protected_content():
    st.title("Welcome to the Protected Area!")
    st.write(f"Hello, {st.session_state.username}!")
    if st.button("Logout"):
        del st.session_state["logged_in"]
        del st.session_state["username"]
        st.experimental_rerun()

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["Login", "Signup"])
if choice == "Login":
    login_page()
elif choice == "Signup":
    signup_page()
