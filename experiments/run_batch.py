"""
Runner multi-univers — backtests Stat Arb
- Télécharge un CSV PAR univers (colonnes strictement = assets de l’univers)
- Option 'mode': 'per_universe' (par défaut) ou 'pooled'
"""
import os, sys
import pandas as pd
import subprocess
import datetime as dt
from pathlib import Path
import yaml

from data.download_prices import download_prices

CONFIG_PATH = Path("config/params.yaml")
RESULTS_DIR = Path("results/batch")
DATA_DIR = Path("data")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Charger config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)

universes = params["universes"]
mode = params.get("mode", "per_universe")  # 'per_universe' | 'pooled'
start = params.get("start", "2024-01-01")
interval = params.get("interval", "1h")

# Normalisation de tickers (ex: SHELL -> SHEL)
ALIAS = {"SHELL": "SHEL"}
def norm_assets(lst): return [ALIAS.get(t, t) for t in lst]

print(f"🧭 Universes détectés : {list(universes.keys())} | mode={mode}")

def run_py(cmd: list[str], env=None) -> int:
    cmd = [sys.executable] + cmd[1:]  # garantir le même venv
    print(f"\n▶️ Lancement : {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)

def nowstamp(): return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

all_results = []

if mode == "pooled":
    # 1 seul CSV avec l’union des actifs
    all_assets = sorted({a for u in universes.values() for a in norm_assets(u["assets"])})
    csv_path = DATA_DIR / "prices_ALL.csv"
    download_prices(all_assets, start=start, interval=interval, path_out=str(csv_path))
    env = os.environ.copy()
    env["UNIVERSE_NAME"] = "ALL"
    env["UNIVERSE_FILE"] = str(csv_path)
    env["UNIVERSE_ASSETS"] = ",".join(all_assets)
    ret = run_py(["python", "-m", "experiments.run_walkforward"], env=env)

    summary_path = Path("results/walkforward_summary.csv")
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        df["universe"] = "ALL"
        df["timestamp"] = nowstamp()
        all_results.append(df)
        print(f"📊 Résultats chargés ({len(df)} lignes)")
else:
    # un CSV par univers, colonnes strictement = assets de l'univers
    for name, u in universes.items():
        print(f"\n=== 🌍 RUNNING UNIVERSE: {name} ===")
        tickers = norm_assets(u["assets"])
        csv_path = DATA_DIR / f"prices_{name}.csv"
        download_prices(tickers, start=start, interval=interval, path_out=str(csv_path))

        env = os.environ.copy()
        env["UNIVERSE_NAME"] = name
        env["UNIVERSE_FILE"] = str(csv_path)
        env["UNIVERSE_ASSETS"] = ",".join(tickers)

        ret = run_py(["python", "-m", "experiments.run_walkforward"], env=env)

        summary_path = Path("results/walkforward_summary.csv")
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["universe"] = name
            df["timestamp"] = nowstamp()
            all_results.append(df)
            print(f"📊 Résultats chargés ({len(df)} lignes)")
        else:
            print("⚠️ Aucun fichier de résultat trouvé, skip.")

# --- Agrégation
if all_results:
    df_all = pd.concat(all_results, ignore_index=True)
    out_path = RESULTS_DIR / f"batch_results_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\n💾 Résultats agrégés sauvegardés → {out_path}")
    print(df_all.groupby("universe")[["Sharpe", "MaxDD", "FinalPnL"]].mean().round(3))
else:
    print("\n⚠️ Aucun résultat à agréger.")
