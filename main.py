# =============================================================================
# main.py — BMS Simulator Entry Point
# =============================================================================
# Launch the BMS Simulator with optional CLI arguments.
#
# Usage:
#   python main.py                          # Default: 1C discharge, passive
#   python main.py --profile udds           # UDDS drive cycle
#   python main.py --mode active            # Active balancing
#   python main.py --soc 0.8               # Start at 80% SOC
#   python main.py --speed 10              # 10x real-time
#   python main.py --headless --duration 3600  # No GUI, log only
# =============================================================================
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import os
import sys

# Ensure project root is on path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="BMS Simulator — 8S8P Li-ion NMC Pack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile", type=str, default="constant_1c",
        choices=[
            "constant_1c", "constant_2c", "constant_half_c",
            "step_hppc", "udds", "cc_cv_charge", "random_walk",
        ],
        help="Load profile for simulation",
    )
    parser.add_argument(
        "--mode", type=str, default="passive",
        choices=["passive", "active"],
        help="Cell balancing mode",
    )
    parser.add_argument(
        "--soc", type=float, default=1.0,
        help="Initial pack SOC [0.0 → 1.0]",
    )
    parser.add_argument(
        "--soc-spread", type=float, default=0.04,
        help="Cell-to-cell SOC imbalance at init [0.0 → 0.1]",
    )
    parser.add_argument(
        "--speed", type=float, default=5.0,
        help="Simulation speed multiplier (1 = real-time)",
    )
    parser.add_argument(
        "--temp", type=float, default=25.0,
        help="Ambient temperature [°C]",
    )
    parser.add_argument(
        "--duration", type=int, default=0,
        help="Simulation duration in seconds (0 = run until closed)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without GUI (log to CSV only)",
    )
    return parser.parse_args()


def run_headless(args):
    """Run simulation without GUI — logs to CSV, prints to console."""
    import time
    from sim_loop import SimLoop

    print("=" * 60)
    print("BMS Simulator — Headless Mode")
    print("=" * 60)
    print(f"  Profile   : {args.profile}")
    print(f"  Balancing : {args.mode}")
    print(f"  SOC init  : {args.soc * 100:.0f}%")
    print(f"  Speed     : {args.speed}x")
    print(f"  Duration  : {args.duration}s")
    print(f"  Ambient   : {args.temp}°C")
    print("=" * 60)

    sim = SimLoop(
        profile        = args.profile,
        balancing_mode = args.mode,
        soc_init       = args.soc,
        soc_spread     = args.soc_spread,
        ambient_temp   = args.temp,
        duration_s     = args.duration if args.duration > 0 else 3600,
        speed          = args.speed,
    )

    def on_tick(state):
        if state.tick % 60 == 0:
            print(
                f"  t={state.time_s:>6.0f}s | "
                f"V={state.pack_voltage:>6.2f}V | "
                f"I={state.pack_current:>6.1f}A | "
                f"SOC={state.pack_soc_ekf*100:>5.1f}% | "
                f"SOH={state.pack_soh*100:>5.1f}% | "
                f"T={state.max_temp_c:>5.1f}°C | "
                f"ΔV={state.delta_v_mv:>5.1f}mV | "
                f"{state.fault_summary}"
            )

    sim.register_callback(on_tick)
    sim.start()

    try:
        while sim._running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")
    finally:
        sim.stop()
        print(f"\n[main] Log saved to: logs/sim_output.csv")


def run_gui(args):
    try:
        from gui.dashboard import launch_dashboard
        launch_dashboard(
            profile        = args.profile,
            balancing_mode = args.mode,
            soc_init       = args.soc,
            speed          = args.speed,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")

def main():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    args = parse_args()

    # Validate SOC range
    if not 0.05 <= args.soc <= 1.0:
        print(f"[Error] --soc must be between 0.05 and 1.0, got {args.soc}")
        sys.exit(1)

    if args.headless:
        run_headless(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()