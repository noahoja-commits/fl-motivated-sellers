"""Streamlit Cloud entry point.

Streamlit Community Cloud defaults the main file path to ``streamlit_app.py``.
This repos dashboard lives in ``app.py``; importing it runs everything,
so the platform default just works without touching the deploy settings.
"""

import app  # noqa: F401  -- importing executes the dashboard module
