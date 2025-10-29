import os
import time
import requests
import pandas as pd
from io import StringIO
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# ============================================================
# ⚙️ INITIALISATION DU DRIVER
# ============================================================

def init_driver(headless=True):
    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver


# ============================================================
# 📡 SCRAPER DUKASCOPY
# ============================================================

def scrape_dukascopy_csv(symbol: str, start="2020-01-01", end="2025-01-01", out_dir="data/dukascopy_selenium"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    driver = init_driver(headless=False)  # ⚠️ mettre False la première fois pour observer
    wait = WebDriverWait(driver, 20)

    try:
        print(f"📡 Ouverture du widget pour {symbol} ...")
        driver.get("https://www.dukascopy.com/swiss/english/marketwatch/historical/")
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(3)

        # --- Fermer le popup cookies s'il apparaît ---
        try:
            print("🍪 Tentative de détection du bandeau cookies...")
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.TAG_NAME, "button"))
            )
            cookie_btns = driver.find_elements(By.TAG_NAME, "button")
            for b in cookie_btns:
                if any(txt in b.text.lower() for txt in ["accept", "agree", "accepter", "ok"]):
                    driver.execute_script("arguments[0].click();", b)
                    print("✅ Cookies acceptés via JavaScript.")
                    time.sleep(2)
                    break
        except Exception as e:
            print(f"(cookie skip) {e}")

        # --- Attendre chargement complet ---
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "input")))
        time.sleep(2)

        # --- Recherche du champ instrument ---
        candidates = driver.find_elements(By.TAG_NAME, "input")
        search_box = None
        for c in candidates:
            name = (c.get_attribute("name") or "").lower()
            aria = (c.get_attribute("aria-label") or "").lower()
            if "instrument" in name or "instrument" in aria:
                search_box = c
                break
        if search_box is None:
            search_box = candidates[0]
        search_box.clear()
        search_box.send_keys(symbol)
        search_box.send_keys(Keys.ENTER)
        print(f"🎯 Instrument sélectionné : {symbol}")
        time.sleep(5)

        # --- Sélection des dates ---
        inputs = driver.find_elements(By.TAG_NAME, "input")
        for inp in inputs:
            name = inp.get_attribute("name").lower()
            if "from" in name:
                inp.clear()
                inp.send_keys(start)
            elif "to" in name:
                inp.clear()
                inp.send_keys(end)
        print(f"🗓️  Période : {start} → {end}")
        time.sleep(1)

        # --- Sélection du format CSV ---
        selects = driver.find_elements(By.TAG_NAME, "select")
        for sel in selects:
            options = sel.find_elements(By.TAG_NAME, "option")
            for opt in options:
                if "csv" in opt.text.lower():
                    opt.click()
                    print("📄 Format CSV sélectionné.")
                    break

        # --- Clic sur Export ---
        btns = driver.find_elements(By.TAG_NAME, "button")
        export_btn = None
        for b in btns:
            if "export" in b.text.lower() or "download" in b.text.lower():
                export_btn = b
                break
        if export_btn is None:
            raise RuntimeError("Bouton Export introuvable")
        export_btn.click()

        print("⏳ Génération du CSV...")
        time.sleep(8)

        # --- Extraction du lien CSV depuis les logs réseau ---
        csv_url = None
        for _ in range(3):  # retry 3 fois
            logs = driver.get_log("performance")
            for entry in logs:
                msg = entry["message"]
                if "freeserv.dukascopy.com" in msg and ".csv" in msg:
                    start_idx = msg.find("https://freeserv")
                    end_idx = msg.find(".csv") + 4
                    csv_url = msg[start_idx:end_idx]
                    break
            if csv_url:
                break
            time.sleep(2)

        if not csv_url:
            print(f"⚠️ Aucun lien CSV trouvé pour {symbol}")
            return None

        print(f"➡️ Lien CSV détecté : {csv_url}")
        csv_data = requests.get(csv_url).text
        df = pd.read_csv(StringIO(csv_data))
        df.columns = [c.strip().capitalize() for c in df.columns]

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date").sort_index()

        csv_path = Path(out_dir) / f"{symbol.replace('/', '_')}.csv"
        df.to_csv(csv_path)
        print(f"✅ Sauvegardé → {csv_path} ({len(df)} lignes)")
        return df

    except Exception as e:
        print(f"❌ Erreur pour {symbol}: {e}")

    finally:
        driver.quit()


# ============================================================
# 🚀 MAIN
# ============================================================

if __name__ == "__main__":
    symbols = ["AAPL.US/USD", "MSFT.US/USD", "GOOG.US/USD"]
    for sym in symbols:
        scrape_dukascopy_csv(sym, start="2020-01-01", end="2025-01-01")
