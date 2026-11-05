"""Shows the sign-in screen and handles the Streamlit login session."""

import streamlit as st

from auth import get_setting, credentials_match


def load_app_settings():
    """Load local settings or Streamlit Cloud secrets."""
    setting_names = ["API_BASE_URL", "APP_USERNAME", "APP_PASSWORD", "API_ACCESS_KEY"]
    settings = {}

    for setting_name in setting_names:
        settings[setting_name] = get_setting(setting_name)

        # Streamlit Cloud keeps deployed secrets in st.secrets.
        if not settings[setting_name]:
            try:
                settings[setting_name] = str(st.secrets.get(setting_name, "")).strip()
            except (FileNotFoundError, AttributeError):
                settings[setting_name] = ""

    if not settings["API_BASE_URL"]:
        settings["API_BASE_URL"] = "http://127.0.0.1:8000"

    settings["API_BASE_URL"] = settings["API_BASE_URL"].rstrip("/")
    return settings


def show_login(settings):
    """Show the login form. Return True after a successful sign-in."""
    username = settings["APP_USERNAME"]
    password = settings["APP_PASSWORD"]
    api_key = settings["API_ACCESS_KEY"]

    if not username or not password or not api_key:
        st.error("Sign-in is not configured yet. Add the login settings and API key first.")
        return False

    if st.session_state.get("is_logged_in", False):
        return True

    left_column, login_column, right_column = st.columns([1, 2, 1])
    with login_column:
        st.subheader("Welcome back")
        st.caption("Sign in to open the customer churn dashboard.")

        with st.form("login_form"):
            entered_username = st.text_input("Username")
            entered_password = st.text_input("Password", type="password")
            submitted = st.form_submit_button(
                "Sign in", type="primary", use_container_width=True
            )

        if submitted:
            if credentials_match(
                entered_username, entered_password, username, password
            ):
                st.session_state["is_logged_in"] = True
                st.rerun()
            else:
                st.error("Those sign-in details don’t match. Please try again.")

    return False


def show_logout_button():
    """Show a button that ends the current browser session's login."""
    if st.sidebar.button("Log out"):
        st.session_state["is_logged_in"] = False
        st.rerun()
