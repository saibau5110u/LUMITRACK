"""
Terminal Live Monitor for LumiTrack (Variant 10.03: Museum Exposure Limiter)
Real-time ANSI terminal dashboard displaying live telemetry, servo position,
cumulative museum exposure, and state machine transitions.
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from host_controller import LumiTrackHostController

def run_terminal_monitor(mode="sim", port=None, scenario="OVEREXPOSURE_RISK"):
    controller = LumiTrackHostController(mode=mode, port=port, scenario=scenario)
    controller.connect()
    
    print("\033[2J\033[H") # Clear terminal
    print("================================================================================")
    print("      LumiTrack: Intelligent Light Mapping & Positioning System (V10.03)        ")
    print("                       Museum Conservation Monitor                              ")
    print("================================================================================")
    print(f" Mode: {mode.upper()} | Scenario: {scenario} | Connection: ACTIVE")
    print(" Press Ctrl+C to exit.\n")
    
    try:
        scan_triggered = False
        start_time = time.time()
        
        while True:
            time.sleep(0.25)
            telem = controller.latest_telemetry
            if not telem:
                continue
                
            # Trigger initial scan after 1.5 seconds
            if not scan_triggered and (time.time() - start_time) > 1.5:
                controller.trigger_scan(profile="FULL")
                scan_triggered = True
                
            state = telem.get("state", "UNKNOWN")
            angle_deg = telem.get("angle_deg", 0)
            angle_id = telem.get("angle_id", 0)
            lux = telem.get("lux", 0.0)
            target_lux = telem.get("target_lux", 500.0)
            cum_lux = telem.get("cum_lux", 0.0)
            safe = "YES (SAFE)" if telem.get("safe", 0) else "ALERT / ACTIVE"
            ovr = "ACTIVE" if telem.get("ovr", 0) else "DISABLED"
            
            # Progress bar for angle (15° to 165°)
            bar_len = 30
            norm_angle = max(0.0, min(1.0, (angle_deg - 15) / (165 - 15)))
            fill_len = int(norm_angle * bar_len)
            angle_bar = "[" + "=" * fill_len + " " * (bar_len - fill_len) + "]"
            
            # Exposure bar
            cum_ratio = min(1.0, cum_lux / 50000.0)
            cum_fill = int(cum_ratio * bar_len)
            cum_bar = "[" + "#" * cum_fill + "-" * (bar_len - cum_fill) + "]"
            
            # Print live dashboard in-place
            sys.stdout.write("\033[6;0H")
            sys.stdout.write(f" [SYSTEM STATE]         : {state:<20}\n")
            sys.stdout.write(f" [SAFETY COMPLIANT]     : {safe:<20}\n")
            sys.stdout.write(f" [MANUAL OVERRIDE]      : {ovr:<20}\n")
            sys.stdout.write("--------------------------------------------------------------------------------\n")
            sys.stdout.write(f" [SERVO ANGLE]          : {angle_deg:>3}° (ID: {angle_id}) {angle_bar}\n")
            sys.stdout.write(f" [MEASURED LIGHT]       : {lux:>6.1f} Lux  (Target Band: {target_lux:>5.1f} Lux)\n")
            sys.stdout.write(f" [CUMULATIVE EXPOSURE]  : {cum_lux:>8.1f} Lux-s / 50,000.0 Lux-s {cum_bar}\n")
            sys.stdout.write("--------------------------------------------------------------------------------\n")
            if controller.latest_scan_result:
                points = controller.latest_scan_result.get("points", 0)
                sys.stdout.write(f" [LATEST SCAN REPORT]   : Completed ({points} sample checkpoints)\n")
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\n[Monitor] Stopping...")
    finally:
        controller.disconnect()

if __name__ == "__main__":
    mode_arg = sys.argv[1] if len(sys.argv) > 1 else "sim"
    port_arg = sys.argv[2] if len(sys.argv) > 2 else "COM3"
    run_terminal_monitor(mode=mode_arg, port=port_arg)
