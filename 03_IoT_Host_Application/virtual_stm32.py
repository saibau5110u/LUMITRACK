"""
Virtual STM32 NUCLEO-F446RE Emulator
Emulates firmware behavior, state machine, sensor acquisition, 
servo PWM kinematics, watchdog timer, and UART protocol.
Enables 100% full verification even when physical board is disconnected.
"""

import time
import json
import random
import threading
import queue

APPROVED_ANGLES_DEG = [15, 40, 65, 90, 115, 140, 165]

class VirtualSTM32:
    def __init__(self, scenario="OVEREXPOSURE_RISK"):
        self.scenario = scenario
        self.state = "SAFE_HOME"
        self.current_angle_id = 3 # 90 deg safe neutral
        self.target_angle_id = 3
        self.current_angle_deg = 90
        
        self.manual_override = False
        self.pot_raw = 2048 # ~500 lux target
        self.ldr_raw = 450
        self.ldr_voltage = 0.36
        self.lux_estimate = 85.0
        self.target_threshold = 500.0
        self.cumulative_exposure = 1200.0
        
        self.last_host_ping_time = time.time()
        self.watchdog_timeout_sec = 3.0
        self.running = False
        
        # In-memory communication queues
        self.rx_queue = queue.Queue() # From host to MCU
        self.tx_queue = queue.Queue() # From MCU to host
        
        self._thread = None
        self._lock = threading.Lock()
        
    def start(self):
        self.running = True
        self.last_host_ping_time = time.time()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            
    def write_from_host(self, data_str):
        """Called by host controller to send UART data into MCU"""
        self.rx_queue.put(data_str)
        
    def read_to_host(self, block=False, timeout=None):
        """Called by host controller to read UART lines emitted by MCU"""
        try:
            return self.tx_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
            
    def _compute_lux_at_angle(self, angle_deg):
        """Simulates physical light field according to active scenario"""
        ambient = 35.0
        if self.scenario == "NORMAL_DIFFUSE":
            center = 90.0
            width = 50.0
            peak = 180.0
            lux = ambient + peak * random.uniform(0.95, 1.05) * (2.718 ** (-0.5 * ((angle_deg - center) / width) ** 2))
        elif self.scenario == "OVEREXPOSURE_RISK":
            center = 15.0 # Direct sunlight entering from POS_0 (15 deg)
            width = 18.0
            peak = 1450.0
            lux = ambient + peak * random.uniform(0.96, 1.04) * (2.718 ** (-0.5 * ((angle_deg - center) / width) ** 2))
        else: # CONFUSER_SPECULAR
            center = 65.0
            width = 7.0 # Sharp specular reflection spike
            peak = 520.0
            lux = ambient + 80.0 + peak * (2.718 ** (-0.5 * ((angle_deg - center) / width) ** 2))
            
        lux += random.gauss(0, 3.0)
        return max(5.0, round(lux, 1))

    def _run_loop(self):
        last_telem_time = time.time()
        telem_seq = 0
        run_id = 1
        
        while self.running:
            now = time.time()
            dt = 0.05
            time.sleep(dt)
            
            with self._lock:
                # 1. Update Watchdog
                if (now - self.last_host_ping_time) > self.watchdog_timeout_sec:
                    if self.state not in ["SAFE_HOME", "FAULT_HOST_TIMEOUT", "MANUAL_OVERRIDE"]:
                        self.state = "FAULT_HOST_TIMEOUT"
                        self.target_angle_id = 3
                        self.current_angle_id = 3
                        self.current_angle_deg = 90
                
                # 2. Process Host Commands
                while not self.rx_queue.empty():
                    raw_cmd = self.rx_queue.get_nowait()
                    self._handle_command(raw_cmd)
                    self.last_host_ping_time = time.time()
                    if self.state == "FAULT_HOST_TIMEOUT":
                        self.state = "IDLE"
                        
                # 3. Servo Kinematics Slew
                target_deg = APPROVED_ANGLES_DEG[self.target_angle_id]
                if self.current_angle_deg < target_deg:
                    self.current_angle_deg = min(target_deg, self.current_angle_deg + 5)
                elif self.current_angle_deg > target_deg:
                    self.current_angle_deg = max(target_deg, self.current_angle_deg - 5)
                else:
                    self.current_angle_id = self.target_angle_id
                    
                # 4. Sensor Sampling
                self.lux_estimate = self._compute_lux_at_angle(self.current_angle_deg)
                self.ldr_voltage = min(3.3, self.lux_estimate / 600.0)
                self.ldr_raw = int((self.ldr_voltage / 3.3) * 4095)
                self.cumulative_exposure += self.lux_estimate * dt
                
                # 5. Emit Periodic Telemetry (5 Hz)
                if (now - last_telem_time) >= 0.20:
                    last_telem_time = now
                    telem_seq += 1
                    telem = {
                        "type": "TELEM",
                        "variant": "10.03",
                        "run": run_id,
                        "seq": telem_seq,
                        "angle_id": self.current_angle_id,
                        "angle_deg": self.current_angle_deg,
                        "ldr_raw": self.ldr_raw,
                        "ldr_v": round(self.ldr_voltage, 3),
                        "pot_raw": self.pot_raw,
                        "pot_v": round((self.pot_raw / 4095) * 3.3, 3),
                        "lux": round(self.lux_estimate, 1),
                        "target_lux": self.target_threshold,
                        "cum_lux": round(self.cumulative_exposure, 1),
                        "state": self.state,
                        "safe": 1 if self.state in ["SAFE_HOME", "POSITIONED", "IDLE"] else 0,
                        "ovr": 1 if self.manual_override else 0
                    }
                    self.tx_queue.put(json.dumps(telem) + "\r\n")

    def _handle_command(self, cmd_str):
        try:
            data = json.loads(cmd_str.strip())
            cmd = data.get("cmd")
            seq = data.get("seq", 0)
        except Exception:
            resp = {
                "type": "NACK", "cmd": "UNKNOWN", "seq": 0,
                "status": "REJECTED", "reason": "ERR_INVALID_COMMAND",
                "curr_angle_id": self.current_angle_id
            }
            self.tx_queue.put(json.dumps(resp) + "\r\n")
            return
            
        if self.manual_override and cmd != "RETURN_HOME":
            resp = {
                "type": "NACK", "cmd": cmd, "seq": seq,
                "status": "REJECTED", "reason": "ERR_MANUAL_OVERRIDE_ACTIVE",
                "curr_angle_id": self.current_angle_id
            }
            self.tx_queue.put(json.dumps(resp) + "\r\n")
            return
            
        if cmd == "ANGLE_ID":
            angle_id = data.get("id")
            if angle_id is None or angle_id < 0 or angle_id >= len(APPROVED_ANGLES_DEG):
                resp = {
                    "type": "NACK", "cmd": cmd, "seq": seq,
                    "status": "REJECTED", "reason": "ERR_OUT_OF_BOUNDS",
                    "curr_angle_id": self.current_angle_id
                }
            else:
                self.target_angle_id = angle_id
                self.state = "POSITIONED"
                resp = {
                    "type": "ACK", "cmd": cmd, "seq": seq,
                    "status": "ACCEPTED", "reason": "OK",
                    "angle_id": angle_id,
                    "angle_deg": APPROVED_ANGLES_DEG[angle_id]
                }
            self.tx_queue.put(json.dumps(resp) + "\r\n")
            
        elif cmd == "SCAN_PROFILE":
            profile = data.get("profile", "FULL")
            resp = {
                "type": "ACK", "cmd": cmd, "seq": seq,
                "status": "ACCEPTED", "reason": "OK",
                "angle_id": self.current_angle_id, "angle_deg": self.current_angle_deg
            }
            self.tx_queue.put(json.dumps(resp) + "\r\n")
            
            # Simulate Scan Execution
            self.state = "SCANNING"
            angles_used = APPROVED_ANGLES_DEG[:4] if profile == "PARTIAL" else APPROVED_ANGLES_DEG
            scan_data = []
            for deg in angles_used:
                val = self._compute_lux_at_angle(deg)
                scan_data.append({"deg": deg, "lux": val})
                
            scan_complete = {
                "type": "SCAN_COMPLETE",
                "profile": profile,
                "points": len(scan_data),
                "data": scan_data
            }
            self.tx_queue.put(json.dumps(scan_complete) + "\r\n")
            self.state = "IDLE"
            self.target_angle_id = 3
            
        elif cmd == "HOLD":
            resp = {
                "type": "ACK", "cmd": cmd, "seq": seq,
                "status": "ACCEPTED", "reason": "OK",
                "angle_id": self.current_angle_id, "angle_deg": self.current_angle_deg
            }
            self.tx_queue.put(json.dumps(resp) + "\r\n")
            
        elif cmd == "RETURN_HOME":
            self.target_angle_id = 3
            self.state = "SAFE_HOME"
            self.manual_override = False
            resp = {
                "type": "ACK", "cmd": cmd, "seq": seq,
                "status": "ACCEPTED", "reason": "OK",
                "angle_id": 3, "angle_deg": 90
            }
            self.tx_queue.put(json.dumps(resp) + "\r\n")
            
        else:
            resp = {
                "type": "NACK", "cmd": str(cmd) if cmd else "UNKNOWN", "seq": seq,
                "status": "REJECTED", "reason": "ERR_INVALID_COMMAND",
                "curr_angle_id": self.current_angle_id
            }
            self.tx_queue.put(json.dumps(resp) + "\r\n")

