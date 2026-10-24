import os
import sys

# Ensure artvision directory and root are on sys.path
API_DIR = os.path.dirname(os.path.abspath(__file__))
ARTVISION_DIR = os.path.abspath(os.path.join(API_DIR, ".."))
ROOT_DIR = os.path.abspath(os.path.join(ARTVISION_DIR, ".."))
for p in [ARTVISION_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Backward-compatibility dynamic exports for FastAPI tests and uvicorn
def __getattr__(name):
    if name in ("app", "health", "predict_endpoint", "similar_endpoint", "get_searcher"):
        from api import server
        return getattr(server, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Execute the ArtVision Streamlit Application if Streamlit is available (e.g. Streamlit Cloud)
try:
    import streamlit as st
    is_streamlit = True
except ImportError:
    is_streamlit = False

if is_streamlit:
    app_py_path = os.path.join(ROOT_DIR, "app.py")
    if os.path.exists(app_py_path):
        with open(app_py_path, "r", encoding="utf-8") as f:
            code = compile(f.read(), app_py_path, "exec")
            exec(code, globals())
