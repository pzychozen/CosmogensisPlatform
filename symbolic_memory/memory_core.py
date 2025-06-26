import sqlite3
import os
import json
import datetime

class RecursiveSymbolicMemoryCore:
    def __init__(self, db_path="data_storage/symbolic_memory.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()

    def create_tables(self):
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_laws (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            law_pattern TEXT,
            source_genomes INTEGER,
            confidence REAL
        )
        """)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS genome_discoveries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            genome_sequence TEXT,
            classification TEXT,
            notes TEXT
        )
        """)
        self.conn.commit()

    def store_law(self, law_pattern, source_genomes, confidence=1.0):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute("""
        INSERT INTO knowledge_laws (timestamp, law_pattern, source_genomes, confidence)
        VALUES (?, ?, ?, ?)
        """, (timestamp, json.dumps(law_pattern), source_genomes, confidence))
        self.conn.commit()

    def store_genome_discovery(self, genome_sequence, classification="unknown", notes=""):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conn.execute("""
        INSERT INTO genome_discoveries (timestamp, genome_sequence, classification, notes)
        VALUES (?, ?, ?, ?)
        """, (timestamp, json.dumps(genome_sequence), classification, notes))
        self.conn.commit()

    def fetch_all_laws(self):
        cursor = self.conn.execute("SELECT * FROM knowledge_laws")
        return cursor.fetchall()

    def fetch_all_genome_discoveries(self):
        cursor = self.conn.execute("SELECT * FROM genome_discoveries")
        return cursor.fetchall()

    def close(self):
        self.conn.close()
