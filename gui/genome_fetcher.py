import sqlite3
import json

def fetch_latest_genomes(db_path, limit=100):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT iteration, genome FROM genomes ORDER BY id DESC LIMIT ?"
    cursor.execute(query, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [(it, json.loads(genome)) for it, genome in rows]
