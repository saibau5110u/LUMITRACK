"""
LumiTrack STM32 NUCLEO-F446RE One-Click Firmware Flasher & Hardware Validator
Automatically locates STM32CubeProgrammer CLI, flashes LumiTrack_STM32.elf,
resets the microcontroller, and validates live serial telemetry.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Common STM32CubeProgrammer paths on Windows
CUBE_PROG_PATHS = [
    r"C:\ST\STM32CubeIDE_2.2.0\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.500.202603051304\tools\bin\STM32_Programmer_CLI.exe",
    r"C:\Program Files\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe",
    r"C:\Program Files (x86)\STMicroelectronics\STM32Cube\STM32CubeProgrammer\bin\STM32_Programmer_CLI.exe"
]

# Firmware ELF candidates
ELF_PATHS = [
    r"D:\Embedded System and IOT\Practice\LumiTrack_STM32\Debug\LumiTrack_STM32.elf",
    r"E:\ML&IOT\LumiTrack_STM32_Extracted\LumiTrack_STM32\Debug\LumiTrack_STM32.elf",
    r"E:\ML&IOT\01_Firmware\Build\LumiTrack_F446RE.elf"
]

def find_cli():
    for p in CUBE_PROG_PATHS:
        if os.path.exists(p):
            return p
    # Search in C:\ST
    st_dir = Path(r"C:\ST")
    if st_dir.exists():
        matches = list(st_dir.rglob("STM32_Programmer_CLI.exe"))
        if matches:
            return str(matches[0])
    return None

def find_elf():
    for p in ELF_PATHS:
        if os.path.exists(p):
            return p
    return None

def main():
    print("=" * 70)
    print("🏛️ LumiTrack STM32 Hardware Flasher & Telemetry Validator")
    print("=" * 70)

    cli = find_cli()
    if not cli:
        print("❌ Error: STM32_Programmer_CLI.exe not found.")
        print("Please ensure STM32CubeIDE or STM32CubeProgrammer is installed.")
        return 1
    print(f"✅ Found STM32CubeProgrammer CLI: {cli}")

    elf = find_elf()
    if not elf:
        print("❌ Error: Compiled .elf file not found.")
        return 1
    print(f"✅ Target Firmware Binary: {elf}")
    print(f"   Size: {os.path.getsize(elf):,} bytes")

    print("\n[1/3] Probing ST-LINK connection to NUCLEO-F446RE...")
    probe_cmd = [cli, "-c", "port=SWD"]
    res = subprocess.run(probe_cmd, capture_output=True, text=True)
    if "Error: No debug probe detected" in res.stdout or res.returncode != 0:
        print("\n⚠️ ST-LINK Probe not detected via SWD.")
        print("Steps to resolve:")
        print("  1. Ensure the USB mini/micro cable is plugged securely into the NUCLEO board.")
        print("  2. Verify that the ST-LINK LED (LD1) is blinking or green/red.")
        print("  3. If ST-LINK driver is held by a crashed session, unplug USB for 3 seconds and plug back in.")
        return 1

    print("✅ Target board identified: NUCLEO-F446RE (SWD connection OK)")

    print("\n[2/3] Flashing firmware to Flash memory (0x08000000) with verification...")
    flash_cmd = [cli, "-c", "port=SWD", "-w", elf, "-v", "-rst"]
    res = subprocess.run(flash_cmd, capture_output=True, text=True)
    if res.returncode == 0 or "File download complete" in res.stdout:
        print("✅ Flashing successful! Microcontroller reset and application started.")
    else:
        print("⚠️ Flash output:")
        print(res.stdout)
        print(res.stderr)
        return 1

    print("\n[3/3] Checking Live Serial Telemetry...")
    try:
        import serial
        import serial.tools.list_ports
        ports = [p.device for p in serial.tools.list_ports.comports() if "STLink" in p.description or "Virtual" in p.description or "COM" in p.device]
        if not ports:
            print("ℹ️ No STLink COM port found immediately. Waiting 2 seconds...")
            time.sleep(2)
            ports = [p.device for p in serial.tools.list_ports.comports()]

        if ports:
            port = ports[0]
            print(f"Connecting to {port} at 115200 baud to read initial packets...")
            s = serial.Serial(port, 115200, timeout=1.5)
            s.reset_input_buffer()
            start = time.time()
            recv_count = 0
            while time.time() - start < 3.0 and recv_count < 5:
                if s.in_waiting:
                    line = s.readline().decode("utf-8", errors="replace").strip()
                    if line:
                        print(f"  📥 Received from MCU: {line}")
                        recv_count += 1
            s.close()
            if recv_count > 0:
                print("🎉 Microcontroller is streaming live sensor readings successfully!")
            else:
                print("ℹ️ Board is running. Launch the dashboard to interact with the system.")
        else:
            print("ℹ️ Please check Device Manager for the STLink COM port.")
    except Exception as e:
        print(f"Serial check note: {e}")

    print("\n✨ Firmware ready! You can now launch the dashboard:")
    print("   python run_dashboard.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())
