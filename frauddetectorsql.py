import sqlite3
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


# cria um banco SQL de teste local (SQLite)
def setup_mock_sqlite_db():
    print("-> Configurando banco de dados SQLite temporário...")
    conn = sqlite3.connect("financial_data.db")
    cursor = conn.cursor()

    # tabela de contas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS accounts (
            account_id INTEGER PRIMARY KEY,
            feat_1 REAL,
            feat_2 REAL,
            feat_3 REAL,
            is_fraud INTEGER
        )
    """)

    # tabela de transações
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            source_id INTEGER,
            target_id INTEGER
        )
    """)

    # Limpa dados antigos se houver
    cursor.execute("DELETE FROM accounts")
    cursor.execute("DELETE FROM transactions")

    # 10 contas fictícias(ids 100 a 109 para testar o remapeamento)
    # contas 105 a 109 serão marcadas como fraude(1)
    np.random.seed(42)
    for idx, account_id in enumerate(range(100, 110)):
        feat_1, feat_2, feat_3 = np.random.rand(3)
        is_fraud = 1 if idx >= 5 else 0
        cursor.execute(
            "INSERT INTO accounts VALUES (?, ?, ?, ?, ?)",
            (account_id, feat_1, feat_2, feat_3, is_fraud)
        )

    # Insere transações(arestas)
    edges = [
        (100, 101), (100, 102), (101, 102), (101, 103), (102, 104), 
        (103, 104), (104, 100), (105, 106), (105, 107), (106, 108), 
        (107, 109), (108, 105), (109, 106), (102, 108), (104, 109)
    ]
    cursor.executemany("INSERT INTO transactions VALUES (?, ?)", edges)

    conn.commit()
    conn.close()


#  leitura dos dados sql
def load_financial_data_from_sql():
    print("-> Lendo dados do banco SQL...")
    
    # Conexão com o banco (Se usar Postgres/MySQL, mude esta linha e os drivers)
    conn = sqlite3.connect("financial_data.db")
    cursor = conn.cursor()

    #  busca de nós (contas)
    cursor.execute("SELECT account_id, feat_1, feat_2, feat_3, is_fraud FROM accounts")
    nodes_raw = cursor.fetchall()
    
    # mapeia ids reais do banco para índices sequenciais iniciando em 0(ex:100->0, 101->1...)
    # o PyTorch Geometric exige obrigatoriamente que as arestas usem índices de 0 até (numero de nodes - 1)
    node_id_map = {row[0]: idx for idx, row in enumerate(nodes_raw)}
    
    x_list = []
    y_list = []
    for row in nodes_raw:
        x_list.append([row[1], row[2], row[3]])  # Características da conta
        y_list.append(row[4])                    # Alvo: 0 (normal) ou 1 (fraude)
        
    x = torch.tensor(x_list, dtype=torch.float)
    y = torch.tensor(y_list, dtype=torch.long)

    # busca de arestas(transações)
    cursor.execute("SELECT source_id, target_id FROM transactions")
    edges_raw = cursor.fetchall()
    
    edge_list = []
    for src, tgt in edges_raw:
        # garante a integridade, adicionando a aresta só se ambos os nós existirem mapeados
        if src in node_id_map and tgt in node_id_map:
            edge_list.append([node_id_map[src], node_id_map[tgt]])
            
    # muda a matriz para o formato exigido pelo PyTorch Geometric: [2, num_edges]
    edges = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    conn.close()
    
    return Data(x=x, edge_index=edges, y=y)


# Adiciona PageRank como feature
def extract_graph_features(data):
    print("-> Calculando PageRank...")

    G = nx.DiGraph()
    G.add_edges_from(data.edge_index.t().numpy())

    pagerank = nx.pagerank(G)
    pr_values = [pagerank.get(i, 0.0) for i in range(data.num_nodes)]

    pr_tensor = torch.tensor(pr_values, dtype=torch.float).view(-1, 1)

    data.x = torch.cat([data.x, pr_tensor], dim=1)

    return data


# Modelo simples de GNN
class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        embeddings = x  # representação interna dos nós

        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), embeddings


# Treino da GNN
def train_model(data):
    print("-> Treinando modelo...")

    learning_rate = 0.01
    epochs = 50
    hidden_channels = 16

    model = FraudGNN(
        in_channels=data.num_node_features,
        hidden_channels=hidden_channels,
        out_channels=2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        out, _ = model(data.x, data.edge_index)

        loss = F.nll_loss(out, data.y)

        loss.backward()
        optimizer.step()

        preds = out.argmax(dim=1)
        f1 = f1_score(data.y.numpy(), preds.numpy(), zero_division=0)

        if epoch % 10 == 0:
            print(f"epoch {epoch} | loss {loss:.4f} | f1 {f1:.4f}")

    return model


# Desenha o grafo com as cores preditas
def visualize_fraud_ring(data, model):
    print("-> Mostrando grafo...")

    model.eval()

    with torch.no_grad():
        out, _ = model(data.x, data.edge_index)
        preds = out.argmax(dim=1).numpy()

    G = nx.DiGraph()
    G.add_edges_from(data.edge_index.t().numpy())

    color_map = ["red" if preds[i] == 1 else "blue" for i in G.nodes()]

    plt.figure(figsize=(10, 8))
    plt.title("Detecção de fraude no grafo com o SQLlite")

    pos = nx.spring_layout(G, seed=42)

    nx.draw(
        G, pos,
        node_color=color_map,
        with_labels=True,
        node_size=800,
        edge_color="gray"
    )

    import matplotlib.patches as mpatches
    plt.legend(handles=[
        mpatches.Patch(color='red', label='fraude'),
        mpatches.Patch(color='blue', label='normal')
    ])

    plt.show()


# Execução do fluxo completo
if __name__ == "__main__":
    # Prepara o banco local SQLite de teste
    setup_mock_sqlite_db()
    
    # Carrega os dados direto do banco via SQL Queries
    graph_data = load_financial_data_from_sql()
    
    # Processamento e extração topológica
    graph_data = extract_graph_features(graph_data)
    
    # Treinamento da Inteligência Artificial
    model = train_model(graph_data)
    
    # Exibição visual do resultado das predições
    visualize_fraud_ring(graph_data, model)