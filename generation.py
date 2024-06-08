import random
import numpy as np


class Graph:
    def __init__(self):
        self.graph_dict = {}

    def add_vertex(self, node):
        if node.position not in self.graph_dict:
            self.graph_dict[node.position] = node

    def add_edge(self, from_node, to_node, weight=0):
        from_node.add_edge(to_node, weight)
        to_node.add_edge(from_node, weight)

    def remove_edge(self, from_node, to_node):
        from_node.remove_edge(to_node)
        to_node.remove_edge(from_node)

    def sorted_keys(self):
        sorted_dict = {k: self.graph_dict[k] for k in sorted(list(self.graph_dict.keys()))}
        self.graph_dict = sorted_dict

class Node:
    def __init__(self, position, id=-1):
        self.position = position
        self.edges = []
        self.id = id
        self.visited = False

    def add_edge(self, to_node, weight):
        self.edges.append([to_node, weight])

    def remove_edge(self, to_node):
        for edge in self.edges:
            if to_node == edge[0]:
                self.edges.remove(edge)


def optimized_generate_graph_random_weight(n, m):
    graph = Graph()
    for id_node, (i, j) in enumerate([(i, j) for i in range(n) for j in range(m)]):
        graph.add_vertex(Node((i,j), id_node))

    for i in range(n):
        for j in range(m):
            if i < n - 1:
                graph.add_edge(graph.graph_dict[(i, j)], graph.graph_dict[(i + 1, j)], random.randint(0, 10))
            if j < m - 1:
                graph.add_edge(graph.graph_dict[(i, j)], graph.graph_dict[(i, j + 1)], random.randint(0, 10))
    return graph


def optimized_generate_graph(n, m):
    graph = Graph()
    for id_node, (i, j) in enumerate([(i, j) for i in range(n) for j in range(m)]):
        graph.add_vertex(Node((i, j), id_node))

    for i in range(n):
        for j in range(m):
            if i < n - 1:
                graph.add_edge(graph.graph_dict[(i, j)], graph.graph_dict[(i + 1, j)], 1)
            if j < m - 1:
                graph.add_edge(graph.graph_dict[(i, j)], graph.graph_dict[(i, j + 1)], 1)
    return graph


def optimized_kruskal(graph):
    def find_parent(node):
        if parent[node] != node:
            parent[node] = find_parent(parent[node])
        return parent[node]

    def merge_sets(node1, node2):
        parent[find_parent(node2)] = find_parent(node1)

    minimum_spanning_tree = []
    edges = [(node, edge[0], edge[1]) for node in graph.graph_dict.values() for edge in node.edges]
    edges.sort(key=lambda x: x[2])
    parent = {node: node for node in graph.graph_dict.values()}

    for node1, node2, weight in edges:
        if find_parent(node1) != find_parent(node2):
            minimum_spanning_tree.append((node1, node2, weight))
            merge_sets(node1, node2)

    graph_res = Graph()
    for node1, node2, weight in minimum_spanning_tree:
        x = graph_res.graph_dict.get(node1.position, Node(node1.position, node1.id))
        y = graph_res.graph_dict.get(node2.position, Node(node2.position, node2.id))
        graph_res.add_vertex(x)
        graph_res.add_vertex(y)
        graph_res.add_edge(x, y, weight)
    graph_res.sorted_keys()
    return graph_res


def generalized_graph_to_maze(graph, n, m):
    maze_size = (2 * n + 1, 2 * m + 1)
    maze = [['W'] * maze_size[1] for _ in range(maze_size[0])]

    for i in range(n):
        for j in range(m):
            maze_x, maze_y = 2 * i + 1, 2 * j + 1
            maze[maze_x][maze_y] = 'P'
            if (i, j) in graph.graph_dict:
                for node, _ in graph.graph_dict[(i, j)].edges:
                    if node.position[0] == i + 1:  # Vertical passage
                        maze[maze_x + 1][maze_y] = 'P'
                    if node.position[1] == j + 1:  # Horizontal passage
                        maze[maze_x][maze_y + 1] = 'P'

    maze[1][0] = 'I'  # Entry at top-left
    maze[2 * n - 1][2 * m] = 'O'  # Exit at bottom-right

    # Convert each row of the maze to a list of strings, each containing one character
    return [[char for char in row] for row in maze]


def create_maze(n, m):
    graph = optimized_generate_graph_random_weight(n, m)
    graph_res = optimized_kruskal(graph)
    return generalized_graph_to_maze(graph_res, n, m)


# Le titre parle de lui-même
def afficher_par_lignes(liste, n):
    if n <= 0:
        print("La valeur de 'n' doit être supérieure à zéro.")
        return

    ligne_actuelle = []
    longueur_ligne_actuelle = 0

    for element in liste:
        if longueur_ligne_actuelle + len(element) <= n:
            ligne_actuelle.append(element)
            longueur_ligne_actuelle += len(element)
        else:
            print(" ".join(map(str, ligne_actuelle)))
            ligne_actuelle = [element]
            longueur_ligne_actuelle = len(element)

    if ligne_actuelle:
        print(" ".join(map(str, ligne_actuelle)))


def recursive_backtracking(graph):
    graph_res = Graph()
    # Initialiser tous les nœuds comme non visités
    visited = {(node.position): False for node in graph.graph_dict.values()}

    # Définir une fonction interne pour le backtracking récursif
    def backtrack(current_node):
        visited[current_node.position] = True  # Marquer le nœud actuel comme visité

        # Mélanger aléatoirement les directions pour garantir un labyrinthe unique à chaque exécution
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            next_pos = (current_node.position[0] + dx, current_node.position[1] + dy)
            if next_pos in graph.graph_dict and not visited[next_pos]:
                # Retirer le "mur" entre le nœud actuel et le voisin
                next_node = graph.graph_dict[next_pos]
                x = graph_res.graph_dict.get(current_node.position, Node(current_node.position, current_node.id))
                y = graph_res.graph_dict.get(next_node.position, Node(next_node.position, next_node.id))
                graph_res.add_vertex(x)
                graph_res.add_vertex(y)
                graph_res.add_edge(x, y, 0)
                backtrack(next_node)  # Continuer le backtracking à partir du voisin

    start_node = random.choice(list(graph.graph_dict.values()))
    backtrack(start_node)
    return graph_res


def recursive_division(graph, width, height):
    def division(graph, i, j, width, height):
        # Width = longueur, Height = hauteur
        # Base case: si la zone à diviser est trop petite, on arrête.
        if width < 2 or height < 2:
            return graph

        # Choix aléatoire de l'orientation du mur à ajouter: horizontale ou verticale.
        # Orientation est choisi basé sur la proportion actuelle et la direction prédéterminée.
        if width > height:
            orientation = 'V'
        elif height > width:
            orientation = 'H'
        else:  # Si le sous-espace est carré, choisissez aléatoirement l'orientation.
            orientation = 'V' if random.choice([True, False]) else 'H'

        # Choix aléatoire du point où placer le mur et l'ouverture.
        if orientation == 'H':  # Mur horizontal donc un passage vertical
            wall = random.randint(i, i + height - 2)  # On choisit aléatoirement la position du mur horizontal
            passage = random.randint(j + 1, j + width - 1)
            for k in range(j, j + width):
                if k != passage:
                    graph.remove_edge(graph.graph_dict[(wall, k)],
                                      graph.graph_dict[(wall + 1, k)])  # On crée le passage
            division(graph, i, j, width, wall - i + 1)  # On continue à faire de même
            division(graph, wall + 1, j, width, i + height - wall - 1)
        else:  # Vertical
            wall = random.randint(j, j + width - 2)
            passage = random.randint(i + 1, i + height - 1)
            for k in range(i, i + height):
                if k != passage:
                    graph.remove_edge(graph.graph_dict[(k, wall)], graph.graph_dict[(k, wall + 1)])  # "Wall"
            division(graph, i, j, wall - j + 1, height)
            division(graph, i, wall + 1, j + width - wall - 1, height)

    division(graph, 0, 0, width, height)
    return graph


def degree_metric(graph):
    deg = 0
    for node in graph.graph_dict.values():
        deg += len(node.edges)
    return deg / len(graph.graph_dict)


# Remarque : le résultat de degree_metric semble dépendre que de n et m taille du labyrinthe et pour n et m fixés, pour kruskal, le résultat semble fixe
def adjacence_matrix(graph):
    adj = []
    for pos_i in graph.graph_dict:
        ligne = []
        for pos_j in graph.graph_dict:
            edge_list = []
            for edge in graph.graph_dict[pos_j].edges:
                edge_list.append(edge[0].position)
            if pos_i in edge_list:
                ligne.append(1)
            else:
                ligne.append(0)
        adj.append(ligne)
    return adj


def shortest_path_metric(
        graph):  # On utilise le fait que notre labyrinthe est un arbre donc le chemin de i à j est unique
    def shortest_path(graph, pos_i, pos_j):
        adj_original = adjacence_matrix(graph)
        adj = adj_original
        n = 1
        while adj[pos_i.id][pos_j.id] == 0:
            adj = np.matmul(adj_original, adj)
            n += 1
        return n

    sp = 0
    for pos_i in graph.graph_dict:
        for pos_j in graph.graph_dict:
            if graph.graph_dict[pos_j].id > graph.graph_dict[pos_i].id:
                sp += shortest_path(graph, graph.graph_dict[pos_i], graph.graph_dict[pos_j])
    n = len(graph.graph_dict)
    return (2 * sp) / (n * (n + 1))


graph = optimized_generate_graph_random_weight(25, 25)
graph = optimized_kruskal(graph)
adj_m = adjacence_matrix(graph)
