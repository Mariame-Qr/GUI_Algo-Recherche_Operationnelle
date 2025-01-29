import tkinter as tk
from tkinter import messagebox
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np


# Classe pour l'ensemble disjoint (utilisée pour Kruskal)
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1


# Fonction pour exécuter Kruskal
def execute_kruskal(num_vertices):
    edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if random.random() < 0.5:
                weight = random.randint(1, 100)
                edges.append((i, j, weight))

    edges.sort(key=lambda x: x[2])
    disjoint_set = DisjointSet(num_vertices)

    mst = []
    total_cost = 0
    for u, v, weight in edges:
        if disjoint_set.find(u) != disjoint_set.find(v):
            disjoint_set.union(u, v)
            mst.append((u, v, weight))
            total_cost += weight

    result = f"Arbre couvrant minimal (MST) :\n"
    for u, v, weight in mst:
        result += f"Arête {u} - {v} avec poids {weight}\n"
    result += f"Coût total de l'MST : {total_cost}"

    messagebox.showinfo("Résultat Kruskal", result)

    # Affichage du graphe
    plot_kruskal_graph(num_vertices, mst)


# Fonction pour afficher le graphe minimal
def plot_kruskal_graph(num_vertices, mst):
    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))
    G.add_weighted_edges_from(mst)

    pos = nx.spring_layout(G)
    edge_labels = {(u, v): f'{w}' for u, v, w in mst}

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, edge_color='gray')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Arbre couvrant minimal (Kruskal)")
    plt.show()


# Fonction pour créer un graphe aléatoire
def create_random_graph(num_vertices, num_edges):
    G = nx.Graph()
    G.add_nodes_from(range(num_vertices))

    while len(G.edges) < num_edges:
        u = random.randint(0, num_vertices - 1)
        v = random.randint(0, num_vertices - 1)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    return G


# Fonction pour Welsh-Powell
def welsh_powell_nodes(graph):
    sorted_nodes = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)
    node_colors = {}
    current_color = 0

    for node in sorted_nodes:
        forbidden_colors = {node_colors[adj] for adj in graph.neighbors(node) if adj in node_colors}

        for color in range(current_color + 1):
            if color not in forbidden_colors:
                node_colors[node] = color
                break
        else:
            current_color += 1
            node_colors[node] = current_color

    return node_colors, current_color + 1


# Fonction pour afficher le graphe coloré
def plot_graph_with_colored_nodes(G, node_colors, chromatic_number):
    pos = nx.spring_layout(G)
    node_color_map = [node_colors[node] for node in G.nodes()]
    cmap = plt.colormaps.get_cmap('tab10')
    nx.draw(G, pos, with_labels=True, node_color=node_color_map, cmap=cmap, node_size=800, font_color='black',
            edge_color='gray')
    plt.title(f"Nombre chromatique X(G) = {chromatic_number}")
    plt.show()


# Fonction pour exécuter Welsh-Powell
def execute_welsh_powell(vertices, edges):
    if edges > vertices * (vertices - 1) // 2:
        messagebox.showerror("Erreur", "Le nombre d'arêtes est trop élevé pour un graphe simple.")
        return

    G = create_random_graph(vertices, edges)
    node_colors, chromatic_number = welsh_powell_nodes(G)

    result_text = f"Nombre chromatique X(G) = {chromatic_number}\n"
    result_text += f"Inégalité Welsh-Powell : {chromatic_number} <= X(G) <= {vertices}"
    messagebox.showinfo("Résultat", result_text)

    plot_graph_with_colored_nodes(G, node_colors, chromatic_number)


# Fenêtres pour les algorithmes
def open_welsh_powell_window():
    input_window = tk.Toplevel(root)
    input_window.title("Welsh-Powell - Entrée")
    input_window.geometry("400x300")

    tk.Label(input_window, text="Nombre de sommets :").pack(pady=10)
    vertices_entry = tk.Entry(input_window)
    vertices_entry.pack(pady=10)

    tk.Label(input_window, text="Nombre d'arêtes :").pack(pady=10)
    edges_entry = tk.Entry(input_window)
    edges_entry.pack(pady=10)

    def on_execute():
        try:
            vertices = int(vertices_entry.get())
            edges = int(edges_entry.get())
            execute_welsh_powell(vertices, edges)
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides.")

    tk.Button(input_window, text="Exécuter", command=on_execute).pack(pady=20)


def open_kruskal_window():
    input_window = tk.Toplevel(root)
    input_window.title("Kruskal - Entrée")
    input_window.geometry("400x200")

    tk.Label(input_window, text="Nombre de sommets :").pack(pady=10)
    vertices_entry = tk.Entry(input_window)
    vertices_entry.pack(pady=10)

    def on_execute():
        try:
            vertices = int(vertices_entry.get())
            execute_kruskal(vertices)
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer une valeur numérique valide.")

    tk.Button(input_window, text="Exécuter", command=on_execute).pack(pady=20)


# Fonction pour exécuter Dijkstra
def execute_dijkstra(num_vertices, start_vertex, end_vertex):
    G = nx.complete_graph(num_vertices)  # Génère un graphe complet
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(1, 100)  # Ajoute des poids aléatoires

    try:
        # Trouver le plus court chemin et sa distance
        length, path = nx.single_source_dijkstra(G, source=start_vertex, target=end_vertex, weight='weight')
        result = f"Chemin le plus court de {start_vertex} à {end_vertex} : {path}\n"
        result += f"Distance totale : {length}"
        messagebox.showinfo("Résultat Dijkstra", result)

        # Affichage du graphe
        plot_dijkstra_graph(G, path)
    except nx.NetworkXNoPath:
        messagebox.showerror("Erreur", f"Aucun chemin trouvé entre {start_vertex} et {end_vertex}.")


# Fonction pour afficher le graphe avec le chemin coloré
def plot_dijkstra_graph(G, path):
    pos = nx.spring_layout(G)
    edge_colors = ['red' if (u in path and v in path and abs(path.index(u) - path.index(v)) == 1) else 'gray' for u, v
                   in G.edges()]
    weights = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, edge_color=edge_colors)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.title("Graphe avec le chemin le plus court")
    plt.show()


# Fenêtre pour Dijkstra
def open_dijkstra_window():
    input_window = tk.Toplevel(root)
    input_window.title("Dijkstra - Entrée")
    input_window.geometry("400x300")

    tk.Label(input_window, text="Nombre de sommets :").pack(pady=10)
    vertices_entry = tk.Entry(input_window)
    vertices_entry.pack(pady=10)

    tk.Label(input_window, text="Sommet de départ :").pack(pady=10)
    start_vertex_entry = tk.Entry(input_window)
    start_vertex_entry.pack(pady=10)

    tk.Label(input_window, text="Sommet d'arrivée :").pack(pady=10)
    end_vertex_entry = tk.Entry(input_window)
    end_vertex_entry.pack(pady=10)

    def on_execute():
        try:
            num_vertices = int(vertices_entry.get())
            start_vertex = int(start_vertex_entry.get())
            end_vertex = int(end_vertex_entry.get())

            if start_vertex < 0 or end_vertex < 0 or start_vertex >= num_vertices or end_vertex >= num_vertices:
                raise ValueError("Les sommets doivent être compris entre 0 et le nombre de sommets - 1.")

            execute_dijkstra(num_vertices, start_vertex, end_vertex)
        except ValueError as e:
            messagebox.showerror("Erreur", str(e))

    tk.Button(input_window, text="Exécuter", command=on_execute).pack(pady=20)






# Fonction pour exécuter l'algorithme Nord-Ouest
def execute_nord_ouest(supply, demand):
    rows, cols = len(supply), len(demand)
    tableau = np.zeros((rows, cols), dtype=int)

    i = j = 0
    while i < rows and j < cols:
        value = min(supply[i], demand[j])
        tableau[i][j] = value
        supply[i] -= value
        demand[j] -= value

        if supply[i] == 0:
            i += 1
        if demand[j] == 0:
            j += 1

    return tableau


# Fonction pour afficher la solution Nord-Ouest
def display_nord_ouest_solution(supply, demand):
    tableau = execute_nord_ouest(supply, demand)
    result_text = "Solution Nord-Ouest :\n" + str(tableau)
    messagebox.showinfo("Résultat Nord-Ouest", result_text)


# Fonction pour exécuter la méthode du Moindre Coût
def execute_moindre_cout(costs, supply, demand):
    rows, cols = len(costs), len(costs[0])
    tableau = np.zeros((rows, cols), dtype=int)

    while np.sum(supply) > 0 and np.sum(demand) > 0:
        min_cost_indices = np.unravel_index(np.argmin(costs, axis=None), costs.shape)
        i, j = min_cost_indices
        value = min(supply[i], demand[j])
        tableau[i][j] = value
        supply[i] -= value
        demand[j] -= value
        costs[i][j] = np.inf  # Empêche de revisiter cette cellule

    return tableau


# Fonction pour afficher la solution du Moindre Coût
def display_moindre_cout_solution(costs, supply, demand):
    tableau = execute_moindre_cout(costs, supply, demand)
    result_text = "Solution Moindre Coût :\n" + str(tableau)
    messagebox.showinfo("Résultat Moindre Coût", result_text)


# Fonction pour exécuter Stepping Stone
def execute_stepping_stone(costs, tableau):
    rows, cols = len(costs), len(costs[0])
    u = [None] * rows
    v = [None] * cols

    u[0] = 0
    for _ in range(rows + cols - 1):
        for i in range(rows):
            for j in range(cols):
                if tableau[i][j] > 0:
                    if u[i] is not None and v[j] is None:
                        v[j] = costs[i][j] - u[i]
                    elif v[j] is not None and u[i] is None:
                        u[i] = costs[i][j] - v[j]

    optimal = True
    for i in range(rows):
        for j in range(cols):
            if tableau[i][j] == 0 and u[i] is not None and v[j] is not None:
                if costs[i][j] < u[i] + v[j]:
                    optimal = False

    result_text = "Solution optimale" if optimal else "Solution non optimale"
    messagebox.showinfo("Résultat Stepping Stone", result_text)


# Fenêtre pour Nord-Ouest
def open_nord_ouest_window():
    input_window = tk.Toplevel(root)
    input_window.title("Nord-Ouest - Entrée")
    input_window.geometry("400x300")

    tk.Label(input_window, text="Capacités des sources (supply) :").pack(pady=5)
    supply_entry = tk.Entry(input_window)
    supply_entry.pack(pady=5)

    tk.Label(input_window, text="Demandes des destinations (demand) :").pack(pady=5)
    demand_entry = tk.Entry(input_window)
    demand_entry.pack(pady=5)

    def on_execute():
        try:
            supply = list(map(int, supply_entry.get().split(',')))
            demand = list(map(int, demand_entry.get().split(',')))
            display_nord_ouest_solution(supply, demand)
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques séparées par des virgules.")

    tk.Button(input_window, text="Exécuter", command=on_execute).pack(pady=20)


# Fenêtre pour Moindre Coût
def open_moindre_cout_window():
    input_window = tk.Toplevel(root)
    input_window.title("Moindre Coût - Entrée")
    input_window.geometry("400x400")

    tk.Label(input_window, text="Matrice des coûts (lignes séparées par point-virgule) :").pack(pady=5)
    costs_entry = tk.Entry(input_window)
    costs_entry.pack(pady=5)

    tk.Label(input_window, text="Capacités des sources (supply) :").pack(pady=5)
    supply_entry = tk.Entry(input_window)
    supply_entry.pack(pady=5)

    tk.Label(input_window, text="Demandes des destinations (demand) :").pack(pady=5)
    demand_entry = tk.Entry(input_window)
    demand_entry.pack(pady=5)

    def on_execute():
        try:
            costs = [list(map(int, row.split(','))) for row in costs_entry.get().split(';')]
            supply = list(map(int, supply_entry.get().split(',')))
            demand = list(map(int, demand_entry.get().split(',')))
            display_moindre_cout_solution(costs, supply, demand)
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    tk.Button(input_window, text="Exécuter", command=on_execute).pack(pady=20)


# Fenêtre pour Stepping Stone
def open_stepping_stone_window():
    input_window = tk.Toplevel(root)
    input_window.title("Stepping Stone - Entrée")
    input_window.geometry("400x400")

    tk.Label(input_window, text="Matrice des coûts (lignes séparées par point-virgule) :").pack(pady=5)
    costs_entry = tk.Entry(input_window)
    costs_entry.pack(pady=5)

    tk.Label(input_window, text="Solution actuelle (lignes séparées par point-virgule) :").pack(pady=5)
    tableau_entry = tk.Entry(input_window)
    tableau_entry.pack(pady=5)

    def on_execute():
        try:
            costs = [list(map(int, row.split(','))) for row in costs_entry.get().split(';')]
            tableau = [list(map(int, row.split(','))) for row in tableau_entry.get().split(';')]
            execute_stepping_stone(costs, tableau)
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs valides.")

    tk.Button(input_window, text="Exécuter", command=on_execute).pack(pady=20)

    # Fonction pour exécuter Bellman-Ford


def execute_bellman_ford(num_vertices, start_vertex, end_vertex):
    G = nx.complete_graph(num_vertices)  # Génère un graphe complet
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(1, 100)  # Ajoute des poids aléatoires

    try:
        # Trouver le plus court chemin et sa distance
        length, path = nx.single_source_bellman_ford(G, source=start_vertex, target=end_vertex, weight='weight')
        result = f"Chemin le plus court de {start_vertex} à {end_vertex} : {path}\n"
        result += f"Distance totale : {length}"
        messagebox.showinfo("Résultat Bellman-Ford", result)

        # Affichage du graphe
        plot_bellman_ford_graph(G, path)
    except nx.NetworkXNoPath:
        messagebox.showerror("Erreur", f"Aucun chemin trouvé entre {start_vertex} et {end_vertex}.")

    # Fonction pour afficher le graphe avec le chemin coloré


def plot_bellman_ford_graph(G, path):
    pos = nx.spring_layout(G)
    edge_colors = ['red' if (u in path and v in path and abs(path.index(u) - path.index(v)) == 1) else 'gray' for u, v
                   in G.edges()]
    weights = nx.get_edge_attributes(G, 'weight')

    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, edge_color=edge_colors)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.title("Graphe avec le chemin le plus court (Bellman-Ford)")
    plt.show()

    # Fenêtre pour Bellman-Ford


def open_bellman_ford_window():
    input_window = tk.Toplevel(root)
    input_window.title("Bellman-Ford - Entrée")
    input_window.geometry("400x300")

    tk.Label(input_window, text="Nombre de sommets :").pack(pady=10)
    vertices_entry = tk.Entry(input_window)
    vertices_entry.pack(pady=10)

    tk.Label(input_window, text="Sommet de départ :").pack(pady=10)
    start_vertex_entry = tk.Entry(input_window)
    start_vertex_entry.pack(pady=10)

    tk.Label(input_window, text="Sommet d'arrivée :").pack(pady=10)
    end_vertex_entry = tk.Entry(input_window)
    end_vertex_entry.pack(pady=10)

    def on_execute():
        try:
            num_vertices = int(vertices_entry.get())
            start_vertex = int(start_vertex_entry.get())
            end_vertex = int(end_vertex_entry.get())

            if start_vertex < 0 or end_vertex < 0 or start_vertex >= num_vertices or end_vertex >= num_vertices:
                raise ValueError("Les sommets doivent être compris entre 0 et le nombre de sommets - 1.")

            execute_bellman_ford(num_vertices, start_vertex, end_vertex)
        except ValueError as e:
            messagebox.showerror("Erreur", str(e))

    tk.Button(input_window, text="Exécuter", command=on_execute).pack(pady=20)






# Fonction pour calculer les coûts réduits
def calculate_reduced_costs(costs, u, v):
    rows, cols = costs.shape
    reduced_costs = np.zeros_like(costs)
    for i in range(rows):
        for j in range(cols):
            reduced_costs[i, j] = costs[i, j] - u[i] - v[j]
    return reduced_costs


# Fonction pour calculer les potentiels u et v
def calculate_potentials(costs, tableau):
    rows, cols = costs.shape
    u = np.full(rows, None)  # Initialiser les potentiels u à None
    v = np.full(cols, None)  # Initialiser les potentiels v à None

    u[0] = 0  # Fixer u[0] à 0 pour commencer

    # Tant qu'il reste des potentiels à calculer
    while None in u or None in v:
        for i in range(rows):
            for j in range(cols):
                if tableau[i][j] > 0:  # Si une case a un flux positif
                    if u[i] is not None and v[j] is None:
                        v[j] = costs[i][j] - u[i]
                    elif v[j] is not None and u[i] is None:
                        u[i] = costs[i][j] - v[j]

    return u, v


# Fonction pour calculer le coût total
def calculate_total_cost(costs, tableau):
    total_cost = np.sum(costs * tableau)
    return total_cost


# Fonction pour exécuter l'algorithme de Potentiel de Metra
def execute_potentiel_metra(costs, tableau):
    rows, cols = costs.shape
    u, v = calculate_potentials(costs, tableau)

    # Calcul des coûts réduits
    reduced_costs = calculate_reduced_costs(costs, u, v)

    # Vérification de la solution optimale
    is_optimal = np.all(reduced_costs >= 0)

    if is_optimal:
        total_cost = calculate_total_cost(costs, tableau)
        result_text = (
            "Solution optimale trouvée avec l'algorithme de Potentiel de Metra.\n\n"
            f"Potentiels u : {u}\n"
            f"Potentiels v : {v}\n\n"
            f"Coûts réduits :\n{reduced_costs}\n\n"
            f"Coût total : {total_cost}"
        )
    else:
        result_text = (
            "La solution n'est pas optimale.\n\n"
            f"Potentiels u : {u}\n"
            f"Potentiels v : {v}\n\n"
            f"Coûts réduits :\n{reduced_costs}\n"
            "Il faut ajuster la solution pour atteindre l'optimalité."
        )

    messagebox.showinfo("Résultat Potentiel de Metra", result_text)


# Fenêtre pour Potentiel de Metra
def open_potentiel_metra_window():
    input_window = tk.Toplevel(root)
    input_window.title("Potentiel de Metra - Entrée")
    input_window.geometry("400x400")

    tk.Label(input_window, text="Matrice des coûts (séparée par des virgules) :").pack(pady=5)
    costs_entry = tk.Entry(input_window)
    costs_entry.pack(pady=5)

    tk.Label(input_window, text="Tableau des flux (séparée par des virgules) :").pack(pady=5)
    tableau_entry = tk.Entry(input_window)
    tableau_entry.pack(pady=5)

    def on_execute():
        try:
            costs = np.array([list(map(int, row.split(','))) for row in costs_entry.get().split(';')])
            tableau = np.array([list(map(int, row.split(','))) for row in tableau_entry.get().split(';')])

            execute_potentiel_metra(costs, tableau)
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques séparées par des virgules.")

    tk.Button(input_window, text="Exécuter", command=on_execute).pack(pady=20)



###############################################################################################################################################



# Algorithme de Ford-Fulkerson
class Graph:
    def __init__(self, graph):
        self.graph = graph
        self.ROW = len(graph)

    def bfs(self, s, t, parent):
        visited = [False] * self.ROW
        queue = [s]
        visited[s] = True

        while queue:
            u = queue.pop(0)

            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        return visited[t]

    def ford_fulkerson(self, source, sink):
        parent = [-1] * self.ROW
        max_flow = 0

        while self.bfs(source, sink, parent):
            path_flow = float("Inf")
            s = sink

            while s != source:
                path_flow = min(path_flow, self.graph[parent[s]][s])
                s = parent[s]

            v = sink
            while v != source:
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

            max_flow += path_flow

        return max_flow


# Interface Tkinter
def open_ford_fulkurson_window():
    def calculate_max_flow():
        try:
            matrix = [[int(x) for x in row.split()] for row in entry_matrix.get("1.0", tk.END).splitlines() if row]
            source = int(entry_source.get())
            sink = int(entry_sink.get())

            g = Graph(matrix)
            max_flow = g.ford_fulkerson(source, sink)

            messagebox.showinfo("Résultat", f"Le flot maximum est : {max_flow}")

        except Exception as e:
            messagebox.showerror("Erreur", "Entrée invalide. Vérifiez la matrice et les sommets.")

    window = tk.Tk()
    window.title("Algorithme de Ford-Fulkerson")

    tk.Label(window, text="Matrice d'adjacence (espaces entre valeurs)").pack()
    entry_matrix = tk.Text(window, height=5, width=50)
    entry_matrix.pack()

    tk.Label(window, text="Sommet source :").pack()
    entry_source = tk.Entry(window)
    entry_source.pack()

    tk.Label(window, text="Sommet puits :").pack()
    entry_sink = tk.Entry(window)
    entry_sink.pack()

    tk.Button(window, text="Calculer Flot Maximum", command=calculate_max_flow).pack()




# Fenêtre principale des algorithmes
def open_algorithms_window():
    algorithms_window = tk.Toplevel(root)
    algorithms_window.title("Algorithmes")
    algorithms_window.geometry("400x400")

    tk.Button(algorithms_window, text="Welsh Powell", command=open_welsh_powell_window).pack(pady=10)
    tk.Button(algorithms_window, text="Kruskal", command=open_kruskal_window).pack(pady=10)
    tk.Button(algorithms_window, text="Dijkstra", command=open_dijkstra_window).pack(pady=10)
    tk.Button(algorithms_window, text="Nord-Ouest", command=open_nord_ouest_window).pack(pady=10)
    tk.Button(algorithms_window, text="Moindre Coût", command=open_moindre_cout_window).pack(pady=10)
    tk.Button(algorithms_window, text="Stepping Stone", command=open_stepping_stone_window).pack(pady=10)
    tk.Button(algorithms_window, text="Bellman-Ford", command=open_bellman_ford_window).pack(pady=10)
    tk.Button(algorithms_window, text="Potentiel Metra", command=open_potentiel_metra_window).pack(pady=10)
    tk.Button(algorithms_window, text="Ford Fulkerson", command=open_ford_fulkurson_window).pack(pady=10)



# Interface principale
root = tk.Tk()
root.title("Interface Principale")
root.geometry("400x200")

tk.Button(root, text="Algorithmes Recherche", command=open_algorithms_window).pack(pady=50)

root.mainloop()
