# Detecção de Fraude com Inteligência Artificial em Grafos e SQL

Este projeto mostra como usar Inteligência Artificial (GNN - Graph Neural Networks) para descobrir contas suspeitas e redes de fraude em transações financeiras. 

O código faz tudo em um só lugar: cria um banco de dados de teste, lê os dados usando comandos SQL, monta a rede (grafo) e treina a IA para achar os fraudadores.

## 📌 O que o código faz?

1. **Cria um banco de dados de teste:** Cria um arquivo de banco de dados local (**SQLite**) com tabelas de contas e transações financeiras fictícias.
2. **Lê os dados com SQL:** Conecta no banco, busca as informações e prepara tudo no formato que a IA precisa (transformando os IDs do banco em índices de 0 a 9).
3. **Calcula o PageRank:** Usa a estrutura da rede para ver quais contas são mais "importantes" ou centrais nas transações e usa isso como uma informação extra para o modelo.
4. **Treina a Inteligência Artificial:** Usa um modelo de Grafos (GNN) para aprender o comportamento das contas e classificar cada uma como *normal* ou *fraude*.
5. **Mostra o resultado visual:** Desenha a rede de transações na tela, pintando as contas normais de **azul** e as fraudes de **vermelho**.

---

## 🛠️ O que você precisa instalado

Para rodar o código, você vai precisar do Python e das seguintes bibliotecas:

```bash
pip install torch torch-geometric networkx numpy matplotlib scikit-learn
