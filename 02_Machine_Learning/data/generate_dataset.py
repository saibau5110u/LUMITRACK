"""
Dataset Generation Script for LumiTrack (Variant 10.03: Museum Exposure Limiter)
Generates physical seed scan datasets adhering strictly to the Learnathon/Hackathon
data dictionary requirements.
"""

import numpy as np
import pandas as pd
import os

APPROVED_ANGLES_DEG = [15, 40, 65, 90, 115, 140, 165]
NUM_POINTS = len(APPROVED_ANGLES_DEG)

def generate_lumi_dataset(n_runs=600, random_state=42):
    np.random.seed(random_state)
    
    records = []
    regimes = ["NORMAL_DIFFUSE", "OVEREXPOSURE_RISK", "CONFUSER_SPECULAR"]
    
    # Generate across multiple independent sessions (for leakage-free group splitting)
    n_sessions = 15
    runs_per_session = n_runs // n_sessions
    
    run_counter = 1
    for sess_id in range(1, n_sessions + 1):
        for _ in range(runs_per_session):
            # Select regime with balanced distribution
            # 40% NORMAL, 35% OVEREXPOSURE, 25% CONFUSER
            regime_idx = np.random.choice([0, 1, 2], p=[0.40, 0.35, 0.25])
            regime = regimes[regime_idx]
            
            # Operator target threshold from potentiometer (lux)
            operator_target = float(np.random.uniform(250.0, 500.0))
            ambient_ref = float(np.random.uniform(20.0, 60.0))
            
            angles = np.array(APPROVED_ANGLES_DEG)
            lux_values = np.zeros(NUM_POINTS)
            
            if regime == "NORMAL_DIFFUSE":
                # Broad, smooth Gaussian centered near middle or window
                center = np.random.uniform(60.0, 120.0)
                width = np.random.uniform(40.0, 70.0)
                peak = np.random.uniform(120.0, 260.0)
                lux_values = ambient_ref + peak * np.exp(-0.5 * ((angles - center) / width) ** 2)
                # Low noise
                lux_values += np.random.normal(0, 5.0, NUM_POINTS)
                condition_id = f"DIFFUSE_GALLERY_AMB_{np.random.randint(1, 6)}"
                
                # In normal conditions, best orientation allows visibility without glare
                # Choose angle with moderate safe light closest to exhibit plane (90 deg)
                rec_angle_id = 3 # Safe neutral
                
            elif regime == "OVEREXPOSURE_RISK":
                # Intense direct beam / aggressive spotlight
                center = np.random.choice([15.0, 40.0, 65.0, 115.0, 140.0, 165.0])
                width = np.random.uniform(15.0, 25.0)
                peak = np.random.uniform(750.0, 1600.0)
                lux_values = ambient_ref + peak * np.exp(-0.5 * ((angles - center) / width) ** 2)
                lux_values += np.random.normal(0, 15.0, NUM_POINTS)
                condition_id = f"DIRECT_SOLAR_SPOT_{np.random.randint(1, 6)}"
                
                # When hazardous light strikes, safe angle turns sensor/shade completely AWAY
                # from peak to protect artifact!
                peak_idx = int(np.argmax(lux_values))
                # Deflect to opposite side or safe closed home
                rec_angle_id = 6 if peak_idx <= 2 else (0 if peak_idx >= 4 else 3)
                
            else: # CONFUSER_SPECULAR
                # Narrow glint from showcase glass, or partial occluding shadow
                center = np.random.uniform(30.0, 150.0)
                width = np.random.uniform(5.0, 12.0) # Very narrow spike
                peak = np.random.uniform(350.0, 650.0)
                baseline = ambient_ref + np.random.uniform(50.0, 100.0)
                lux_values = baseline + peak * np.exp(-0.5 * ((angles - center) / width) ** 2)
                # Add sensor jitter
                lux_values += np.random.normal(0, 10.0, NUM_POINTS)
                condition_id = f"CABINET_SPECULAR_GLINT_{np.random.randint(1, 5)}"
                
                # Confuser: don't panic to extreme angles; maintain safe neutral (POS_3)
                rec_angle_id = 3
                
            lux_values = np.clip(lux_values, 5.0, 2000.0)
            
            # Cumulative rate: integral of lux across positions
            trapezoidal_sum = np.trapezoid(lux_values, angles) if hasattr(np, 'trapezoid') else np.trapz(lux_values, angles)
            cum_rate = float(trapezoidal_sum / (165 - 15))
            
            record = {
                "run_id": run_counter,
                "variant_id": "10.03",
                "session_id": f"SESS_{sess_id:02d}",
                "condition_id": condition_id,
                "operator_target_lux": round(operator_target, 1),
                "ambient_ref_lux": round(ambient_ref, 1),
                "lux_pos0_15deg": round(lux_values[0], 1),
                "lux_pos1_40deg": round(lux_values[1], 1),
                "lux_pos2_65deg": round(lux_values[2], 1),
                "lux_pos3_90deg": round(lux_values[3], 1),
                "lux_pos4_115deg": round(lux_values[4], 1),
                "lux_pos5_140deg": round(lux_values[5], 1),
                "lux_pos6_165deg": round(lux_values[6], 1),
                "cum_rate_lux_avg": round(cum_rate, 1),
                "regime_class": regime_idx,
                "regime_name": regime,
                "recommended_angle_id": rec_angle_id,
                "recommended_angle_deg": APPROVED_ANGLES_DEG[rec_angle_id]
            }
            records.append(record)
            run_counter += 1
            
    df = pd.DataFrame(records)
    return df

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate_lumi_dataset(n_runs=600, random_state=42)
    csv_path = os.path.join(out_dir, "seed_scan_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Successfully generated {len(df)} runs to {csv_path}")
    print("Class distribution:")
    print(df["regime_name"].value_counts())
