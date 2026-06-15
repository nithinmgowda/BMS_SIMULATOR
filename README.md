# 🔋 BMS Simulator

A real-time Battery Management System simulator for an 8S8P NMC lithium-ion pack, built entirely in Python. Models cell-level electrochemistry, State of Charge estimation via Extended Kalman Filter, passive/active cell balancing, thermal behaviour, and multi-level fault detection — all visualised in a live PyQt5 dashboard.

---

## Screenshot

> *(Add a screenshot of your dashboard here)*

---

## Features

### Pack Modelling
- **8S8P NMC lithium-ion pack** — 64 cells total, configurable series/parallel topology
- Per-cell voltage, internal resistance, and capacity modelling with realistic spread
- Cell-level temperature simulation with thermal coupling

### State Estimation
- **Extended Kalman Filter (EKF)** for real-time State of Charge (SOC) estimation
- State of Health (SOH) tracking over cycle life
- Pack energy accounting in Wh

### Load Profiles
| Profile | Description |
|---|---|
| `constant_1c` | Constant 1C discharge |
| `constant_2c` | Constant 2C discharge |
| `constant_half_c` | Constant 0.5C discharge |
| `step_hppc` | Hybrid Pulse Power Characterisation test |
| `udds` | Urban Dynamometer Driving Schedule cycle |
| `cc_cv_charge` | Constant Current / Constant Voltage charging |
| `random_walk` | Stochastic load for stress testing |

### Cell Balancing
- **Passive balancing** — dissipative shunt resistor approach, triggers at ΔV > 20 mV
- **Active balancing** — energy-redistribution approach, triggers at ΔV > 10 mV
- Live per-cell balancing activity tracking

### Fault Detection
Multi-level fault system (INFO → WARNING → CRITICAL → EMERGENCY) covering:
- Overvoltage / Undervoltage per cell
- Overtemperature
- Pack-level anomalies

### Dashboard (PyQt5 + Matplotlib)
- **Cell Voltage Bar Chart** — per series-group mean voltage, colour-coded by state
- **SOC Gauge** — half-donut arc gauge, colour shifts green → yellow → red
- **SOH Trend** — rolling time-series plot with end-of-life threshold marker
- **Temperature Heatmap** — 2D per-cell heatmap (series × parallel)
- **Balancing Activity Plot** — ΔV history with active balancing event markers
- **Fault Log** — live scrolling log with severity-coded status indicator
- **Simulation controls** — profile selector, balancing mode toggle, pause/resume, speed up to 50×

---

## Project Structure

```
bms-simulator/
├── gui/
│   └── dashboard.py       # PyQt5 dashboard — all panels, controls, header
├── sim_loop.py            # Simulation engine and state management
├── config.py              # Pack configuration constants
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- PyQt5
- Matplotlib
- NumPy

### Installation

```bash
git clone https://github.com/your-username/bms-simulator.git
cd bms-simulator
pip install -r requirements.txt
```

### Running the Simulator

```bash
python gui/dashboard.py
```

Or launch programmatically with custom parameters:

```python
from gui.dashboard import launch_dashboard

launch_dashboard(
    profile="udds",
    balancing_mode="active",
    soc_init=0.9,
    speed=10.0,
)
```

### Configuration

Edit `config.py` to change pack topology or cell parameters:

```python
NUM_CELLS_SERIES   = 8      # Series groups
NUM_CELLS_PARALLEL = 8      # Parallel cells per group
CELL_MAX_VOLTAGE_V = 4.2    # NMC upper cutoff
CELL_MIN_VOLTAGE_V = 2.8    # NMC lower cutoff
PACK_NOMINAL_VOLTAGE_V = 29.6
```

---

## Requirements

```
PyQt5>=5.15
matplotlib>=3.7
numpy>=1.24
```

---

## Roadmap

- [ ] Degradation model (capacity fade vs. cycle count)
- [ ] Thermal runaway simulation
- [ ] SIL (Software-in-the-Loop) testing against real BMS firmware
- [ ] Data logging to CSV / HDF5
- [ ] Configurable cell chemistry (LFP, NCA)
- [ ] Web dashboard alternative (Dash / Streamlit)

---

## Key Technical Notes

**EKF tuning** — process noise covariance Q is the most sensitive parameter. Too low and the filter lags load transients; too high and it amplifies sensor noise. Start with `Q = 1e-4` for the SOC state and tune from there.

**Rendering performance** — all Matplotlib panels use `draw_idle()` instead of `draw()` on updates to avoid blocking the Qt event loop. The figure canvases are pre-sized with explicit `figsize` and `subplots_adjust` so they render correctly at startup without waiting for a resize event.

**Balancing thresholds** — passive balancing at 20 mV looks reasonable in simulation but dissipates meaningful energy at scale. Active balancing at 10 mV is more efficient but adds circuit complexity.

---

## License

MIT License — see `LICENSE` for details.

---

## Author

**Your Name**  
Electrical Engineering Student  
[LinkedIn](https://linkedin.com/in/your-profile) · [GitHub](https://github.com/your-username)
