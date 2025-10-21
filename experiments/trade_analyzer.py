import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")

def analyze_trades(equity_path: Path, mode="WF", trade_log_dir="results/trades"):
    """
    Analyse les trades individuels et les périodes de saut dans le PnL.
    - equity_path : chemin du fichier d'équity agrégée (ex: portfolio_equity_wf.csv)
    - trade_log_dir : dossier contenant les logs ou csv de trades par paire
    """
    print(f"🔍 Analyse des trades pour {mode}...")

    # === Lecture de l'équity agrégée ===
    equity = pd.read_csv(equity_path, parse_dates=["Date"], index_col="Date")["Portfolio"]
    equity_diff = equity.diff().fillna(0)

    # === Identification des "sauts" anormaux ===
    threshold = equity_diff.std() * 5
    jumps = equity_diff[abs(equity_diff) > threshold]
    print(f"⚠️ {len(jumps)} sauts identifiés (> {threshold:.4f})")

    # === Lecture optionnelle des logs de trades ===
    trade_dir = Path(trade_log_dir)
    if not trade_dir.exists():
        print("⚠️ Aucun dossier de logs de trades trouvé.")
    else:
        trade_files = list(trade_dir.glob("*.csv"))
        if trade_files:
            print(f"📂 {len(trade_files)} fichiers de trades trouvés.")
            all_trades = pd.concat(
                [pd.read_csv(f).assign(Pair=f.stem) for f in trade_files], ignore_index=True
            )
            # Exemple de métriques
            print("=== 📊 Statistiques globales ===")
            print(all_trades[["PnL", "Duration", "EntryTime"]].describe())

            # Histogramme des PnL individuels
            plt.figure(figsize=(8,4))
            all_trades["PnL"].hist(bins=50)
            plt.title(f"Distribution du PnL par trade ({mode})")
            plt.xlabel("PnL (€)")
            plt.ylabel("Nombre de trades")
            plt.tight_layout()
            plt.show()

    # === Visualisation des sauts ===
    plt.figure(figsize=(10, 5))
    plt.plot(equity.index, equity, label="Equity totale")
    plt.scatter(jumps.index, equity.loc[jumps.index], color="red", label="Sauts détectés")
    plt.title(f"Analyse des sauts d'équity ({mode})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return jumps


if __name__ == "__main__":
    wf_equity_path = RESULTS_DIR / "portfolio_equity_wf.csv"
    jumps = analyze_trades(wf_equity_path, mode="WF")
    print("\n=== Dates des sauts majeurs ===")
    print(jumps.sort_values(ascending=False).head(10))
