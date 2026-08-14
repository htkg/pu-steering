"""Extract tradeoff data from results/*/summary.json and print as Python dicts.

Usage:
  python plots/extract_plot_data.py
  python plots/extract_plot_data.py --results_dir results
"""

import argparse
import json
import os
import sys


def load_summary(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    base = args.results_dir

    # --- Single-class ---
    classes = {
        "Plane": ("plane", "0.001"),
        "Frog": ("frog", "1.0"),
        "Cat": ("cat", "0.001"),
        "Ship": ("ship", "1.0"),
    }
    scales = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
    lams = [0.5, 1.0, 1.5, 2.0]

    print("# SafeDenoiser sweep (80 seeds)")
    print("SD = {")
    for cls_name, (cls_key, _pu_lam) in classes.items():
        rate, fid, kl, mpd = [], [], [], []
        for sc in scales:
            s = load_summary(os.path.join(base, f"{cls_key}_sd_{sc}", "summary.json"))
            if s is None:
                print(f"    # WARNING: missing {cls_key}_sd_{sc}", file=sys.stderr)
                rate.append(None)
                fid.append(None)
                kl.append(None)
                mpd.append(None)
            else:
                rate.append(round(s["target_rate"] * 100, 1))
                fid.append(round(s["fid"], 2))
                kl.append(round(s["kl"], 3))
                mpd.append(round(s["mpd"], 3))
        print(f'    "{cls_name}": {{')
        print(f'        "scale": {scales},')
        print(f'        "rate": {rate},')
        print(f'        "FID": {fid},')
        print(f'        "KL": {kl},')
        print(f'        "MPD": {mpd},')
        print("    },")
    print("}")

    print()
    print("# Ours class-specific best reg (80 seeds)")
    for cls_name, (cls_key, pu_lam) in classes.items():
        print(f"# {cls_name}: pu_lam={pu_lam}")
    print("Ours = {")
    for cls_name, (cls_key, _pu_lam) in classes.items():
        rate, fid, kl, mpd = [], [], [], []
        # lam=0 is SD 0.3
        s = load_summary(os.path.join(base, f"{cls_key}_sd_0.3", "summary.json"))
        if s:
            rate.append(round(s["target_rate"] * 100, 1))
            fid.append(round(s["fid"], 2))
            kl.append(round(s["kl"], 3))
            mpd.append(round(s["mpd"], 3))
        for lam in lams:
            s = load_summary(os.path.join(base, f"{cls_key}_ours_lam{lam}", "summary.json"))
            if s is None:
                print(f"    # WARNING: missing {cls_key}_ours_lam{lam}", file=sys.stderr)
                rate.append(None)
                fid.append(None)
                kl.append(None)
                mpd.append(None)
            else:
                rate.append(round(s["target_rate"] * 100, 1))
                fid.append(round(s["fid"], 2))
                kl.append(round(s["kl"], 3))
                mpd.append(round(s["mpd"], 3))
        print(f'    "{cls_name}": {{')
        print(f'        "lam": [0, {", ".join(str(l) for l in lams)}],')
        print(f'        "rate": {rate},')
        print(f'        "FID": {fid},')
        print(f'        "KL": {kl},')
        print(f'        "MPD": {mpd},')
        print("    },")
    print("}")

    # --- Vehicles ---
    print()
    print("# Vehicles multi-class (80 seeds)")
    v_scales = scales
    rate, fid, kl, mpd = [], [], [], []
    for sc in v_scales:
        s = load_summary(os.path.join(base, f"vehicles_sd_{sc}", "summary.json"))
        if s is None:
            print(f"# WARNING: missing vehicles_sd_{sc}", file=sys.stderr)
            rate.append(None)
            fid.append(None)
            kl.append(None)
            mpd.append(None)
        else:
            rate.append(round(s["target_rate"] * 100, 1))
            fid.append(round(s["fid"], 2))
            kl.append(round(s["kl"], 3))
            mpd.append(round(s["mpd"], 4))
    print("SD_vehicles = {")
    print(f'    "scale": {v_scales},')
    print(f'    "rate": {rate},')
    print(f'    "FID": {fid},')
    print(f'    "KL": {kl},')
    print(f'    "MPD": {mpd},')
    print("}")

    print()
    v_ours_configs = [
        ("0.5", "vehicles_ours_lam0.5"),
        ("1.0", "vehicles_ours_lam1.0"),
        ("1.5", "vehicles_ours_lam1.5"),
        ("2.0", "vehicles_ours_lam2.0_pl5"),
    ]
    v_lams, v_rate, v_fid, v_kl, v_mpd = [], [], [], [], []
    for lam, dirname in v_ours_configs:
        s = load_summary(os.path.join(base, dirname, "summary.json"))
        v_lams.append(float(lam))
        if s is None:
            print(f"# WARNING: missing {dirname}", file=sys.stderr)
            v_rate.append(None)
            v_fid.append(None)
            v_kl.append(None)
            v_mpd.append(None)
        else:
            v_rate.append(round(s["target_rate"] * 100, 1))
            v_fid.append(round(s["fid"], 2))
            v_kl.append(round(s["kl"], 3))
            v_mpd.append(round(s["mpd"], 4))
    print("Ours_vehicles = {")
    print(f'    "lam": {v_lams},')
    print('    "label": ["$\\lambda$=0.5", "$\\lambda$=1.0", "$\\lambda$=1.5", "$\\lambda$=2.0\\n(pl=5)"],')
    print(f'    "rate": {v_rate},')
    print(f'    "FID": {v_fid},')
    print(f'    "KL": {v_kl},')
    print(f'    "MPD": {v_mpd},')
    print("}")


if __name__ == "__main__":
    main()
