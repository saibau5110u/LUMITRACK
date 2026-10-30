"""
Universal Dashboard Launcher for LumiTrack (02_Machine_Learning subfolder)
Can be run directly from 02_Machine_Learning:
    python run_dashboard.py
"""

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_SCRIPT = os.path.join(PROJECT_ROOT, "03_IoT_Host_Application", "dashboard_app.py")

print(f"[Dashboard Launcher] Launching LumiTrack Streamlit Dashboard...")
print(f"  Target: {DASHBOARD_SCRIPT}")
cmd = [sys.executable, "-m", "streamlit", "run", DASHBOARD_SCRIPT]
subprocess.run(cmd)
