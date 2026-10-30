"""
Packet Codec and Protocol Formatter
LumiTrack (Variant 10.03: Museum Exposure Limiter)
Strictly adheres to Parent Problem 10 canonical command contract:
ANGLE_ID | SCAN_PROFILE | HOLD | RETURN_HOME
"""

import json
import time

CANONICAL_COMMANDS = ["ANGLE_ID", "SCAN_PROFILE", "HOLD", "RETURN_HOME"]

def encode_host_command(cmd_type, angle_id=None, profile=None, seq_num=1, model_ver="1.0", confidence=1.0, expiry_ms=5000):
    """
    Encodes a canonical host command packet into JSON string for UART transmission.
    """
    if cmd_type not in CANONICAL_COMMANDS:
        raise ValueError(f"Invalid canonical command '{cmd_type}'. Must be one of {CANONICAL_COMMANDS}")
        
    payload = {
        "cmd": cmd_type,
        "seq": int(seq_num),
        "model_ver": str(model_ver),
        "conf": round(float(confidence), 3),
        "exp": int(expiry_ms)
    }
    
    if cmd_type == "ANGLE_ID":
        if angle_id is None:
            raise ValueError("ANGLE_ID command requires 'angle_id' (0 to 6)")
        payload["id"] = int(angle_id)
    elif cmd_type == "SCAN_PROFILE":
        payload["profile"] = str(profile or "FULL")
        
    packet_str = json.dumps(payload) + "\r\n"
    return packet_str

def decode_mcu_message(line):
    """
    Decodes an incoming line from the STM32 MCU.
    Supports JSON packets (TELEM, ACK, NACK, SCAN_COMPLETE)
    as well as ASCII/CSV telemetry (DATA,time,angle,adc,lux) and statuses.
    """
    if not line:
        return None
        
    line_clean = line.strip()
    if not line_clean:
        return None
        
    # 1. Try standard JSON
    try:
        data = json.loads(line_clean)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 2. Parse CSV format: DATA,<time_ms>,<angle>,<adc>,<lux>
    if line_clean.startswith("DATA,"):
        parts = line_clean.split(",")
        if len(parts) >= 5:
            try:
                time_ms = float(parts[1])
                angle = float(parts[2])
                adc = int(float(parts[3]))
                lux = float(parts[4])
                approved_angles = [15, 40, 65, 90, 115, 140, 165]
                angle_id = min(range(len(approved_angles)), key=lambda i: abs(approved_angles[i] - angle))
                return {
                    "type": "TELEM",
                    "variant": "10.03",
                    "seq": int(time_ms // 100),
                    "angle_id": angle_id,
                    "angle_deg": int(angle),
                    "ldr_raw": adc,
                    "ldr_v": round(adc * 3.3 / 4095.0, 3),
                    "lux": round(lux, 1),
                    "target_lux": 500.0,
                    "cum_lux": round(lux * (time_ms / 1000.0), 1),
                    "state": "IDLE",
                    "safe": 1,
                    "ovr": 0
                }
            except Exception:
                pass

    # 3. Parse ASCII status and responses
    if line_clean.startswith("SCAN_DONE"):
        return {"type": "SCAN_COMPLETE", "status": line_clean}
    elif line_clean in ["SCAN_ACCEPTED", "STOP_ACCEPTED", "STOPPED", "HOME_DONE", "REFINE_DONE"]:
        return {"type": "ACK", "reason": "OK", "msg": line_clean}
    elif line_clean.startswith("ERROR"):
        return {"type": "NACK", "reason": line_clean}
    elif line_clean in ["BH1750_READY", "LUMITRACK_READY"] or "WARNING" in line_clean:
        return {"type": "STATUS", "status": line_clean}

    # Non-JSON debug output
    return {"type": "RAW_DEBUG", "message": line_clean}
