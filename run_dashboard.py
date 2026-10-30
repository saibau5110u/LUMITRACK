"""
Universal Dashboard Launcher for LumiTrack
Can be run from ANY directory:
    python run_dashboard.py
Automatically uses 'python -m streamlit run' to launch the web dashboard.
"""

import os
import sys
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) in ["02_Machine_Learning", "03_IoT_Host_Application", "04_Verification_and_Tests"]:
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
else:
    PROJECT_ROOT = CURRENT_DIR

DASHBOARD_SCRIPT = os.path.join(PROJECT_ROOT, "03_IoT_Host_Application", "dashboard_app.py")

print(f"[Dashboard Launcher] Launching LumiTrack Streamlit Dashboard...")
print(f"  Target: {DASHBOARD_SCRIPT}")
cmd = [sys.executable, "-m", "streamlit", "run", DASHBOARD_SCRIPT]
subprocess.run(cmd)
