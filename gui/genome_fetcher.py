import sqlite3
import json

def create_tables(db_path):
    # Connect to the database and create the table if it doesn't exist
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = """
    CREATE TABLE IF NOT EXISTS genomes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        experiment TEXT,
        iteration INTEGER,
        genome TEXT,
        parameters TEXT
    )
    """
    cursor.execute(query)
    conn.commit()
    conn.close()

def fetch_latest_genomes(db_path, limit=100):
    # Ensure the table exists before attempting to fetch data
    create_tables(db_path)

    # Connect to the database and fetch the latest genomes
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT iteration, genome FROM genomes ORDER BY id DESC LIMIT ?"
    cursor.execute(query, (limit,))
    rows = cursor.fetchall()
    conn.close()

    # Return the fetched genomes
    return [(it, json.loads(genome)) for it, genome in rows]

