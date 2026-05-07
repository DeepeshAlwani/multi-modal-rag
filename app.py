import streamlit as st
import requests
import json
import time

st.set_page_config(page_title="Multi-Modal RAG", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOKEN_KEY = "rag_session_token"
API_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# LocalStorage helpers via a tiny bidirectional HTML component
# ---------------------------------------------------------------------------

def _ls_get(key: str) -> str | None:
    """
    Inject a one-shot JS snippet that reads localStorage and writes the value
    into a hidden Streamlit text component via postMessage.
    We use st.session_state as a cache so we only do the async round-trip once
    per browser session.
    """
    # We can't do true sync JS→Python in Streamlit, so we embed a self-contained
    # component that reads the value and immediately posts it back via a query
    # param redirect trick.  Instead, we use the simpler pattern:
    # store the token in a hidden text_input whose default is injected by JS.
    pass  # see _render_auth_bridge below


def _render_auth_bridge():
    """
    Renders an invisible iframe (same-origin) that redirects parent window
    with token as query parameter if token exists in localStorage.
    Only rendered when no token in query params to avoid infinite redirect loops.
    """
    st.iframe(
        "/static/auth-bridge.html",
        height=1,
        width=1,
    )


def _save_token_to_browser(token: str):
    """Write a token into the browser's localStorage via same-origin iframe."""
    st.iframe(
        f"/static/save-token.html?token={token}",
        height=1,
        width=1,
    )


def _clear_token_from_browser():
    """Remove the token from the browser's localStorage via same-origin iframe."""
    st.iframe(
        "/static/clear-token.html",
        height=1,
        width=1,
    )


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "token" not in st.session_state:
    st.session_state.token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "repo_indexed" not in st.session_state:
    st.session_state.repo_indexed = False
if "messages" not in st.session_state:
    st.session_state.messages = []
# Flag so we only attempt auto-login once per Streamlit session
if "_checked_stored_token" not in st.session_state:
    st.session_state._checked_stored_token = False

# ---------------------------------------------------------------------------
# Persistent login: read token from localStorage on first load
# ---------------------------------------------------------------------------
# Check if token came via query param (from auth-bridge redirect after page reload)
query_params = st.query_params
stored_token = query_params.get("auth_token", "")

if stored_token and not st.session_state._checked_stored_token:
    # Validate the stored token against the API
    try:
        resp = requests.get(
            f"{API_URL}/repo_info",
            headers={"Authorization": f"Bearer {stored_token}"},
            timeout=5,
        )
        if resp.status_code != 401:
            # Token is still valid — restore session
            st.session_state.token = stored_token
            st.session_state.authenticated = True
            st.session_state.user_email = st.session_state.get("_stored_email", "")
            # Check if they already have a repo indexed
            if resp.status_code == 200:
                st.session_state.repo_indexed = True
    except Exception:
        pass  # API not reachable; stay logged out
    st.session_state._checked_stored_token = True

# Render hidden iframe for future page reloads (only if no token in query params and not yet checked)
if not stored_token and not st.session_state._checked_stored_token:
    _render_auth_bridge()
    st.session_state._checked_stored_token = True


# ---------------------------------------------------------------------------
# Helper: call logout endpoint and clear everything
# ---------------------------------------------------------------------------
def do_logout():
    if st.session_state.token:
        try:
            requests.post(
                f"{API_URL}/logout",
                headers={"Authorization": f"Bearer {st.session_state.token}"},
                timeout=5,
            )
        except Exception:
            pass
    _clear_token_from_browser()
    st.session_state.authenticated = False
    st.session_state.token = None
    st.session_state.user_email = None
    st.session_state.repo_indexed = False
    st.session_state.messages = []
    st.session_state._checked_stored_token = False
    st.rerun()


# ---------------------------------------------------------------------------
# Login / Register screen
# ---------------------------------------------------------------------------
if not st.session_state.authenticated:
    st.title("🔐 Multi-Modal RAG Assistant")

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                try:
                    response = requests.post(
                        f"{API_URL}/login",
                        json={"email": email, "password": password},
                    )
                    if response.status_code == 200:
                        data = response.json()
                        token = data["token"]
                        st.session_state.token = token
                        st.session_state.authenticated = True
                        st.session_state.user_email = email
                        st.session_state._checked_stored_token = True
                        # Persist token to browser localStorage
                        _save_token_to_browser(token)
                        # Check if user already has a repo indexed from a previous session
                        try:
                            ri = requests.get(
                                f"{API_URL}/repo_info",
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=5,
                            )
                            if ri.status_code == 200:
                                st.session_state.repo_indexed = True
                        except Exception:
                            pass
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    with tab2:
        with st.form("register_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Register")

            if submitted:
                if password != confirm:
                    st.error("Passwords don't match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    try:
                        response = requests.post(
                            f"{API_URL}/register",
                            json={"email": email, "password": password},
                        )
                        if response.status_code == 200:
                            st.success("Registration successful! Please login.")
                        else:
                            st.error(response.json().get("detail", "Registration failed"))
                    except Exception as e:
                        st.error(f"Connection error: {e}")

# ---------------------------------------------------------------------------
# Main app (after login)
# ---------------------------------------------------------------------------
else:
    with st.sidebar:
        st.header(f"👤 {st.session_state.user_email or 'User'}")

        if not st.session_state.repo_indexed:
            st.subheader("📦 Step 1: Add GitHub Repository")
            repo_url = st.text_input(
                "Public GitHub Repo URL",
                placeholder="https://github.com/username/repo.git",
            )

            if st.button("Clone & Index Repository"):
                if repo_url:
                    with st.spinner("Cloning and indexing repository… This may take a few minutes."):
                        try:
                            response = requests.post(
                                f"{API_URL}/clone_repo",
                                headers={"Authorization": f"Bearer {st.session_state.token}"},
                                json={"repo_url": repo_url},
                            )
                            if response.status_code == 200:
                                st.session_state.repo_indexed = True
                                st.success("Repository indexed successfully!")
                                st.rerun()
                            else:
                                st.error(response.json().get("detail", "Failed"))
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Please enter a repository URL")
        else:
            st.success("✅ Repository indexed")
            if st.button("📂 Change Repository"):
                # Tell the API to clear the active repo for this user
                try:
                    requests.post(
                        f"{API_URL}/clear_repo",
                        headers={"Authorization": f"Bearer {st.session_state.token}"},
                        timeout=5,
                    )
                except Exception:
                    pass
                st.session_state.repo_indexed = False
                st.session_state.messages = []
                st.rerun()

        st.markdown("---")
        st.caption("🤖 RAG System | Code + Diagrams | Local LLM")

    # Main chat area
    st.title("💬 Multi-Modal RAG Chat")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if st.session_state.repo_indexed:
        # Sidebar repo stats
        try:
            response = requests.get(
                f"{API_URL}/repo_info",
                headers={"Authorization": f"Bearer {st.session_state.token}"},
            )
            if response.status_code == 200:
                info = response.json()
                with st.sidebar:
                    st.markdown(f"**📊 Repository Stats:** {info['total_functions']} functions indexed")
                    if info.get("repo_url"):
                        st.caption(f"🔗 {info['repo_url']}")
                    with st.expander("📝 View Functions"):
                        for func in info['functions'][:10]:
                            st.code(
                                f"{func['name']} ({func['file'].split('/')[-1]})",
                                language="python",
                            )
        except Exception:
            pass

        if prompt := st.chat_input("Ask about the codebase..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                collected = [""]

                def token_generator():
                    try:
                        resp = requests.post(
                            f"{API_URL}/query/stream",
                            headers={"Authorization": f"Bearer {st.session_state.token}"},
                            json={"question": prompt, "session_id": "default"},
                            stream=True,
                            timeout=(10, 120),
                        )
                        for line in resp.iter_lines(chunk_size=1):
                            if line and line.startswith(b"data: "):
                                try:
                                    data = json.loads(line[6:])
                                    if "token" in data:
                                        collected[0] += data["token"]
                                        yield data["token"]
                                    elif "error" in data:
                                        yield f"\n\n⚠️ Error: {data['error']}"
                                except json.JSONDecodeError:
                                    pass
                    except Exception as e:
                        yield f"\n\n⚠️ Connection error: {e}"

                st.write_stream(token_generator())
                st.session_state.messages.append(
                    {"role": "assistant", "content": collected[0]}
                )
    else:
        st.info("👈 Please enter a GitHub repository URL in the sidebar to get started.")

# ---------------------------------------------------------------------------
# Logout (always visible when authenticated)
# ---------------------------------------------------------------------------
with st.sidebar:
    if st.session_state.authenticated:
        if st.button("🚪 Logout"):
            do_logout()