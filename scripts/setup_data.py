"""
Downloads raw Wyscout data from figshare into data/raw/.
Source: https://figshare.com/collections/Soccer_match_event_dataset/4415000/5

Run: python scripts/setup_data.py
"""
import os
import json
import urllib.request
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
EVENTS_DIR  = RAW_DIR / "events"
MATCHES_DIR = RAW_DIR / "matches"

EVENTS_DIR.mkdir(parents=True, exist_ok=True)
MATCHES_DIR.mkdir(parents=True, exist_ok=True)

# Figshare download URLs (version 5)
FIGSHARE_BASE = "https://figshare.com/ndownloader/files"

FILES = {
    # Events
    EVENTS_DIR  / "events_Italy.json":                "15073868",
    EVENTS_DIR  / "events_Germany.json":              "14464685",
    EVENTS_DIR  / "events_England.json":              "14464685",  # placeholder
    EVENTS_DIR  / "events_Spain.json":                "15073882",
    EVENTS_DIR  / "events_France.json":               "15073876",
    EVENTS_DIR  / "events_World_Cup.json":            "14464685",  # placeholder
    EVENTS_DIR  / "events_European_Championship.json":"14464685",  # placeholder
    # Matches
    MATCHES_DIR / "matches_Italy.json":               "15073712",
    MATCHES_DIR / "matches_Germany.json":             "14464685",  # placeholder
    MATCHES_DIR / "matches_England.json":             "14464685",  # placeholder
    MATCHES_DIR / "matches_Spain.json":               "14464685",  # placeholder
    MATCHES_DIR / "matches_France.json":              "14464685",  # placeholder
    MATCHES_DIR / "matches_World_Cup.json":           "14464685",  # placeholder
    MATCHES_DIR / "matches_European_Championship.json":"14464685", # placeholder
    # Reference files
    RAW_DIR / "players.json":       "15073721",
    RAW_DIR / "teams.json":         "15073697",
    RAW_DIR / "coaches.json":       "15073724",
    RAW_DIR / "referees.json":      "15073730",
    RAW_DIR / "playerank.json":     "15074030",
    RAW_DIR / "eventid2name.csv":   "14464725",
    RAW_DIR / "tags2name.csv":      "14464724",
}


class DownloadProgress(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url, dest):
    with DownloadProgress(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)


if __name__ == "__main__":
    print("NOTE: Update the figshare file IDs in this script with the correct ones")
    print("from: https://figshare.com/collections/Soccer_match_event_dataset/4415000/5\n")

    for dest, file_id in FILES.items():
        if dest.exists():
            print(f"  [skip] {dest.name} already exists")
            continue
        url = f"{FIGSHARE_BASE}/{file_id}"
        print(f"  Downloading {dest.name}...")
        try:
            download(url, dest)
        except Exception as e:
            print(f"  [ERROR] {e}")
