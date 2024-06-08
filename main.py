import numpy as np
from generation import *
from courbe import tracer_regression_lineaire, tracer_histogramme
import matplotlib.pyplot as plt
import os
import sys

sys.setrecursionlimit(2000)
def matrice_de_transition(adjacence):
    n = adjacence.shape[0]
    somme_lignes = np.sum(adjacence, axis=1)

    # Identifier les états absorbants et transitoires
    absorbants = np.where(somme_lignes == 0)[0]
    transitoires = np.where(somme_lignes != 0)[0]

    # Création des sous-matrices Q, R, 0 et I
    Q = adjacence[np.ix_(transitoires, transitoires)]
    R = adjacence[np.ix_(transitoires, absorbants)]
    I = np.eye(len(absorbants))
    zero = np.zeros((len(absorbants), len(transitoires)))

    # Normalisation de la matrice Q
    somme_lignes_transitoires = somme_lignes[transitoires][:, None]
    Q = Q / somme_lignes_transitoires

    # Construction de la matrice de transition P
    haut = np.hstack((Q, R))
    bas = np.hstack((zero, I))
    P = np.vstack((haut, bas))
    P[n-1][n-1] = 1
    return P

def extraire_Q(matrice_transition):
    n = matrice_transition.shape[0]

    # Identifier les indices des états transitoires et absorbants
    absorbants = np.where(np.diag(matrice_transition) == 1)[0]
    transitoires = np.where(np.diag(matrice_transition) != 1)[0]

    # Extraire la sous-matrice Q
    Q = matrice_transition[np.ix_(transitoires, transitoires)]

    return Q

def calculer_matrice_fondamentale(Q):
    n = Q.shape[0] # Taille de la matrice Q
    I = np.eye(n) # Matrice identité I de la même taille que Q
    N = np.linalg.inv(I - Q) # Matrice fondamentale N = (I - Q)^{-1}
    return N

def calculer_vecteur_temps_absorption(N):
    n = N.shape[0]
    vecteur_1 = np.ones((n, 1)) # vecteur colonne de 1 de taille n
    t = np.dot(N, vecteur_1) # t = N * vecteur_1
    return t[0]

def lab_temps_transition(maze):
    L = np.array(adjacence_matrix(maze))
    T = matrice_de_transition(L)
    Q = extraire_Q(T)
    N = calculer_matrice_fondamentale(Q)
    return calculer_vecteur_temps_absorption(N)[0]

def dijkstra(matrice_adjacence, depart, arrivee):
    # Nombre de sommets dans le graphe
    n = matrice_adjacence.shape[0]

    # Distances minimum de depart à chaque sommet
    distances = np.full(n, np.inf)
    distances[depart] = 0

    # Tableau pour suivre les sommets dont le plus court chemin est déjà trouvé
    visite = np.zeros(n, dtype=bool)

    for _ in range(n):
        # Choisir le sommet avec la distance minimum non visité
        min_distance = np.inf
        u = -1
        for sommet in range(n):
            if not visite[sommet] and distances[sommet] < min_distance:
                min_distance = distances[sommet]
                u = sommet

        # Marquer le sommet choisi comme visité
        visite[u] = True

        # Mise à jour des distances des voisins non visités
        for v in range(n):
            if matrice_adjacence[u][v] > 0 and not visite[v]:
                nouvelle_distance = distances[u] + matrice_adjacence[u][v]
                if nouvelle_distance < distances[v]:
                    distances[v] = nouvelle_distance

    # Retourne la distance au sommet d'arrivée
    return distances[arrivee]


'''EVOLUTION AVEC LA TAILLE
n = 10
for t in liste_taille:
    L = []
    for i in range(n): # On calcule une moyenne sur n labyrinthes générés
        maze = optimized_generate_graph_random_weight(t,t)
        maze = optimized_kruskal(maze)
        L.append(lab_temps_transition(maze))
    liste_tps.append(sum(L)/n)

plt.figure()
plt.plot(liste_taille, liste_tps, marker='o', linestyle='-', color='b')
plt.xlabel('Taille du labyrinthe')
plt.ylabel('Temps d\'absorption moyen')
plt.title('Évolution du temps d\'absorption en fonction de la taille du labyrinthe')
plt.grid(True)
plt.show()
'''

'''COMPARAISON AVEC DIJSKTRA
liste_pcc = []
liste_tps = []
for i in range(5, 10,5):
    t1, t2 = [], []
    for j in range(10):
        maze = optimized_generate_graph_random_weight(i, i)
        maze = optimized_kruskal(maze)
        L = np.array(adjacence_matrix(maze))
        T = matrice_de_transition(L)
        Q = extraire_Q(T)
        N = calculer_matrice_fondamentale(Q)
        t1.append(calculer_vecteur_temps_absorption(N)[0])
        t2.append(dijkstra(L, 0, i - 1))
    liste_tps.append(sum(t1)/len(t1))
    liste_pcc.append(sum(t2)/len(t2))

tracer_regression_lineaire(liste_pcc,liste_tps)
'''

'''
#REPARTITION DES VALEURS POUR UNE TAILLE DONNEE
n = 20 #taille labyrinthe
p = 1000 #Nombre de valeurs
L = []
for i in range(p):
    maze = optimized_generate_graph_random_weight(n,n)
    maze = optimized_kruskal(maze)
    L.append(lab_temps_transition(maze))
tracer_histogramme(L)
'''

# Paramètres
n = 100  # Nombre d'expériences
liste_taille = [i for i in range(2, 52, 2)]


# Fonction pour générer et enregistrer les graphiques
def generate_and_save_plot(titles, file_names, generate_maze_functions):
    for title, file_name, generate_maze_function in zip(titles, file_names, generate_maze_functions):
        liste_tps = []
        for t in liste_taille:
            L = []
            for i in range(n):  # On calcule une moyenne sur n labyrinthes générés
                maze = optimized_generate_graph_random_weight(t, t)
                if title == 'Recursive Division':
                    maze = generate_maze_function(maze, t, t)  # Passe les dimensions en plus
                else:
                    maze = generate_maze_function(maze)
                L.append(lab_temps_transition(maze))
            liste_tps.append(sum(L) / n)
            print(title + str(t))

        plt.figure()
        plt.plot(liste_taille, liste_tps, marker='o', linestyle='-', color='b')
        plt.xlabel('Taille du labyrinthe')
        plt.ylabel('Temps d\'absorption moyen')
        plt.title(title)
        plt.grid(True)

        # Enregistrer le graphique dans un fichier
        output_file = os.path.join(os.path.dirname(__file__), file_name)
        plt.savefig(output_file)


# Noms des méthodes et fichiers associés
titles = ['Kruskal', 'Recursive Backtracking', 'Recursive Division']
file_names = ['kruskal_stats.png', 'recursive_backtracking_stats.png', 'recursive_division_stats.png']
generate_maze_functions = [optimized_kruskal, recursive_backtracking, recursive_division]

# Générer et enregistrer les graphiques
generate_and_save_plot(titles, file_names, generate_maze_functions)