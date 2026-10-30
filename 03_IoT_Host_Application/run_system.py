"""
LumiTrack System Launcher
Unified runner for Terminal Monitor, Web Dashboard, Simulation, and Live Hardware.

Usage:
  python run_system.py --mode sim             (Simulated closed-loop test)
  python run_system.py --mode monitor         (Terminal CLI live dashboard)
  python run_system.py --mode gui             (Streamlit web dashboard)
  python run_system.py --mode live --port COM3 (Live hardware on specified COM port)
"""

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="LumiTrack System Launcher")
    parser.add_argument("--mode", choices=["sim", "monitor", "gui", "live"], default="sim",
                        help="Launch mode: 'sim', 'monitor', 'gui', or 'live'")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port for live mode (e.g. COM3)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--scenario", choices=["OVEREXPOSURE_RISK", "NORMAL_DIFFUSE", "CONFUSER_SPECULAR"],
                        default="OVEREXPOSURE_RISK", help="Lighting condition scenario for simulation")
    
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.mode == "sim":
        print(f"[Launcher] Running Simulated Closed-Loop ML Test (Scenario: {args.scenario})...")
        from host_controller import LumiTrackHostController
        import time
        ctrl = LumiTrackHostController(mode="sim", scenario=args.scenario)
        ctrl.connect()
        time.sleep(1.0)
        ctrl.trigger_scan(profile="FULL")
        time.sleep(3.5)
        ctrl.return_home()
        time.sleep(1.0)
        ctrl.disconnect()
        print("\n[Launcher] Simulation completed successfully.")
        
    elif args.mode == "monitor":
        print("[Launcher] Starting Terminal CLI Monitor...")
        from terminal_monitor import run_terminal_monitor
        run_terminal_monitor(mode="sim", scenario=args.scenario)
        
    elif args.mode == "gui":
        print("[Launcher] Launching Streamlit Web Dashboard...")
        dashboard_script = os.path.join(script_dir, "dashboard_app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_script])
        
    elif args.mode == "live":
        print(f"[Launcher] Starting Live Hardware Connection on {args.port}...")
        from terminal_monitor import run_terminal_monitor
        run_terminal_monitor(mode="live", port=args.port)

if __name__ == "__main__":
    main()
