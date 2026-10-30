# LumiTrack Dataset Data Dictionary (Variant 10.03: Museum Exposure Limiter)

## 1. Overview
This dataset contains physical angular light scans recorded by the LumiTrack STM32 NUCLEO-F446RE light mapper. Each row represents one independent scan trial across the 7 approved discrete servo angles.

## 2. Schema and Field Descriptions

| Column Name | Data Type | Units | Description | Permitted Range |
| :--- | :--- | :--- | :--- | :--- |
| `run_id` | Integer | - | Unique incremental run identifier | 1 to N |
| `variant_id` | String | - | Hackathon problem variant identifier | Fixed `"10.03"` |
| `session_id` | String | - | Laboratory / acquisition session identifier | `SESS_01` to `SESS_15` |
| `condition_id` | String | - | Experimental setup / lighting source fixture descriptor | String |
| `operator_target_lux` | Float | Lux | Target exposure threshold configured via 10k potentiometer | 50.0 to 1500.0 |
| `ambient_ref_lux` | Float | Lux | Ambient/dark reference level measured at safe zero plane | 10.0 to 100.0 |
| `lux_pos0_15deg` | Float | Lux | Calibrated light intensity measured at POS_0 (15°) | 0.0 to 2000.0 |
| `lux_pos1_40deg` | Float | Lux | Calibrated light intensity measured at POS_1 (40°) | 0.0 to 2000.0 |
| `lux_pos2_65deg` | Float | Lux | Calibrated light intensity measured at POS_2 (65°) | 0.0 to 2000.0 |
| `lux_pos3_90deg` | Float | Lux | Calibrated light intensity measured at POS_3 (90°, Safe Home) | 0.0 to 2000.0 |
| `lux_pos4_115deg` | Float | Lux | Calibrated light intensity measured at POS_4 (115°) | 0.0 to 2000.0 |
| `lux_pos5_140deg` | Float | Lux | Calibrated light intensity measured at POS_5 (140°) | 0.0 to 2000.0 |
| `lux_pos6_165deg` | Float | Lux | Calibrated light intensity measured at POS_6 (165°) | 0.0 to 2000.0 |
| `cum_rate_lux_avg` | Float | Lux | Average angular flux across the 7 scan positions | 0.0 to 2000.0 |
| `regime_class` | Integer | - | Numeric regime label: 0, 1, or 2 | `[0, 1, 2]` |
| `regime_name` | String | - | Light regime name: `NORMAL_DIFFUSE`, `OVEREXPOSURE_RISK`, `CONFUSER_SPECULAR` | Categorical |
| `recommended_angle_id` | Integer | - | Canonical discrete safe angle index recommended by expert policy | 0 to 6 |
| `recommended_angle_deg`| Integer | Degrees | Angle in degrees corresponding to `recommended_angle_id` | 15, 40, 65, 90, 115, 140, 165 |

## 3. Light Regimes Defined

- **Class 0: `NORMAL_DIFFUSE`**
  - Smooth daylight/gallery ambient light. Well within conservation tolerances.
  - Safe Action: Exhibit remains viewable; servo parks at neutral plane (`POS_3`, 90°).

- **Class 1: `OVEREXPOSURE_RISK`**
  - Dangerous high-intensity direct solar ray or hot spotlight exceeding cumulative exposure threshold.
  - Operationally Costly / Hazardous Class!
  - Safe Action: Clamps shade/louver away from peak angle or reverts to safe neutral closed state to prevent irreversible photochemical damage.

- **Class 2: `CONFUSER_SPECULAR`**
  - Specular glass glint or localized shadow artifact. High localized spike but low total flux.
  - Nearest-Confuser Class to prevent naive threshold false alarms.
  - Safe Action: Maintains exhibit orientation (`POS_3`) without unnecessary panic closure.
