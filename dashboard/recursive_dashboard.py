import dash
from dash import dcc, html
import plotly.graph_objs as go
import sqlite3
import json

class RecursiveDashboard:
    def __init__(self, db_path="data_storage/genome_database.db"):
        self.db_path = db_path
        self.app = dash.Dash(__name__)
        self.layout()

    def fetch_genomes(self):
        conn = sqlite3.connect(self.db_path)
        query = "SELECT id, iteration, genome FROM genomes ORDER BY id DESC LIMIT 100"
        cursor = conn.execute(query)
        data = []
        for row in cursor:
            genome_seq = json.loads(row[2])
            data.append({
                'id': row[0],
                'iteration': row[1],
                'genome': genome_seq
            })
        conn.close()
        return data

    def layout(self):
        self.app.layout = html.Div([
            html.H1("Recursive Species Evolution Dashboard"),
            dcc.Graph(id='genome-distribution'),
            dcc.Interval(id='interval-update', interval=5000, n_intervals=0)
        ])

        @self.app.callback(
            dash.dependencies.Output('genome-distribution', 'figure'),
            [dash.dependencies.Input('interval-update', 'n_intervals')]
        )
        def update_graph(n):
            data = self.fetch_genomes()

            if not data:
                return go.Figure()

            genome_lengths = [len(item['genome']) for item in data]
            iterations = [item['iteration'] for item in data]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=iterations,
                y=genome_lengths,
                mode='markers+lines',
                name='Genome Length'
            ))

            fig.update_layout(
                xaxis_title='Iteration',
                yaxis_title='Genome Length',
                title='Recursive Genome Evolution',
                template='plotly_dark'
            )
            return fig

    def run(self):
        self.app.run_server(debug=True, port=8050)
