# Arquivo: ag_functions.py
import numpy as np

def create_individual():
    """Gera um cromossomo com hiperparâmetros válidos para RandomForest."""
    return [
        np.random.randint(10, 200),    # n_estimators (int)
        np.random.randint(1, 20),      # max_depth (int)
        np.random.uniform(0.01, 0.5)   # min_samples_split (float)
    ]

def crossover(parent1, parent2):
    """Realiza o cruzamento de um ponto entre dois pais."""
    point = np.random.randint(1, 3)
    child = parent1[:point] + parent2[point:]
    return child

def mutate(individual, prob=0.1):
    """Aplica mutação garantindo que o tipo do hiperparâmetro seja respeitado."""
    if np.random.rand() < prob:
        gene_index = np.random.randint(0, 3)
        new_ind = create_individual()
        individual[gene_index] = new_ind[gene_index]
    return individual