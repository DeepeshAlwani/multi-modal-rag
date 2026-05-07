import streamlit as st
import requests
import json
import time

st.set_page_config(page_title="Multi-Modal RAG", page_icon="🤖", layout="wide")

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "token" not in st.session_state:
    st.session_state.token = None
if "repo_indexed" not in st.session_state:
    st.session_state.repo_indexed = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"

# Login/Register screen
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
                        f"{st.session_state.api_url}/login",
                        json={"email": email, "password": password}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.token = data["token"]
                        st.session_state.authenticated = True
                        st.session_state.user_email = email
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
                            f"{st.session_state.api_url}/register",
                            json={"email": email, "password": password}
                        )
                        if response.status_code == 200:
                            st.success("Registration successful! Please login.")
                        else:
                            st.error(response.json().get("detail", "Registration failed"))
                    except Exception as e:
                        st.error(f"Connection error: {e}")

# Main app (after login)
else:
    # Sidebar
    with st.sidebar:
        st.header(f"👤 {st.session_state.user_email}")
        
        # GitHub repo input (if not indexed yet)
        if not st.session_state.repo_indexed:
            st.subheader("📦 Step 1: Add GitHub Repository")
            repo_url = st.text_input("Public GitHub Repo URL", 
                                     placeholder="https://github.com/username/repo.git")
            
            if st.button("Clone & Index Repository"):
                if repo_url:
                    with st.spinner("Cloning and indexing repository... This may take a few minutes."):
                        try:
                            response = requests.post(
                                f"{st.session_state.api_url}/clone_repo",
                                headers={"Authorization": f"Bearer {st.session_state.token}"},
                                json={"repo_url": repo_url}
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
                st.session_state.repo_indexed = False
                st.session_state.messages = []
                st.rerun()
        
        st.markdown("---")
        st.caption("🤖 RAG System | Code + Diagrams | Local LLM")
    
    # Main chat area
    st.title("💬 Multi-Modal RAG Chat")
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Chat input
    if st.session_state.repo_indexed:
        try:
            response = requests.get(
                f"{st.session_state.api_url}/repo_info",
                headers={"Authorization": f"Bearer {st.session_state.token}"}
            )
            if response.status_code == 200:
                info = response.json()
                with st.sidebar:
                    st.markdown(f"**📊 Repository Stats:** {info['total_functions']} functions indexed")
                    with st.expander("📝 View Functions"):
                        for func in info['functions'][:10]:
                            st.code(f"{func['name']} ({func['file'].split('/')[-1]})", language="python")
        except:
            pass
        if prompt := st.chat_input("Ask about the codebase..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get streaming response
            with st.chat_message("assistant"):
                # Use a list as a mutable container so the generator can accumulate text
                # (nonlocal fails here because Streamlit re-runs the whole script as module-level code)
                collected = [""]

                def token_generator():
                    try:
                        resp = requests.post(
                            f"{st.session_state.api_url}/query/stream",
                            headers={"Authorization": f"Bearer {st.session_state.token}"},
                            json={"question": prompt, "session_id": "test"},
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
                st.session_state.messages.append({"role": "assistant", "content": collected[0]})
    
    else:
        st.info("👈 Please enter a GitHub repository URL in the sidebar to get started.")

# Logout button in sidebar
with st.sidebar:
    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.token = None
        st.session_state.repo_indexed = False
        st.session_state.messages = []
        st.rerun()