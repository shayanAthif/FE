"""Quick database inspection script."""
import sqlite3

conn = sqlite3.connect("database/hidden_risk.db")
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)
for t in tables:
    n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: {n} rows")
conn.close()

