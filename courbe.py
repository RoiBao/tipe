import numpy as np
import matplotlib.pyplot as plt

def tracer_regression_lineaire(x, y):
    # Conversion des listes en arrays numpy si nécessaire
    x = np.array(x)
    y = np.array(y)

    # Calcul de la régression linéaire
    coefficients = np.polyfit(x, y, 1)
    m, b = coefficients
    print("Pente (m):", m)
    print("Ordonnée à l'origine (b):", b)

    # Calcul du coefficient de corrélation de Pearson
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2
    print("Coefficient de corrélation (R^2):", r_squared)

    # Création de la ligne de régression
    y_reg = m * x + b

    # Tracé des points de données et de la ligne de régression
    plt.scatter(x, y, color='blue', label='Données')
    plt.plot(x, y_reg, color='red', label=f'Ligne de régression y = {m:.2f}x + {b:.2f}')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # Afficher les coefficients sur le graphique
    # Utilisation de coordonnées relatives pour positionner le texte
    text_x = 0.05  # 5% de la largeur du graphique
    text_y = 0.95  # 95% de la hauteur du graphique
    plt.text(text_x, text_y, f'Pente = {m:.2f}\nR² = {r_squared:.2f}',
             fontsize=12, color='black', transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', alpha=0.5))

    plt.show()


def tracer_histogramme(valeurs):
    """
    Fonction pour tracer l'histogramme d'une liste de valeurs.

    :param valeurs: Liste des valeurs pour lesquelles l'histogramme sera tracé
    """
    # Création de l'histogramme
    plt.hist(valeurs, bins='auto', color='blue', alpha=0.7)

    # Titre et étiquettes des axes
    plt.xlabel('Temps d\'absorption')
    plt.ylabel('Fréquence')

    # Affichage du graphique
    plt.show()