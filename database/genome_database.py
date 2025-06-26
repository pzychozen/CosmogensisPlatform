import sqlite3
import os
import json
import time

class RecursiveGenomeDatabase:
    def __init__(self, db_path="data_storage/genome_database.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.create_tables()

    def create_tables(self):
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
        self.conn.execute(query)
        self.conn.commit()

    def store_genome(self, experiment_name, iteration, genome_sequence, parameters=None):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        genome_json = json.dumps(genome_sequence)
        param_json = json.dumps(parameters) if parameters else None

        query = """
        INSERT INTO genomes (timestamp, experiment, iteration, genome, parameters)
        VALUES (?, ?, ?, ?, ?)
        """
        self.conn.execute(query, (timestamp, experiment_name, iteration, genome_json, param_json))
        self.conn.commit()

    def fetch_all_genomes(self):
        query = "SELECT * FROM genomes"
        cursor = self.conn.execute(query)
        return cursor.fetchall()

    def close(self):
        self.conn.close()
