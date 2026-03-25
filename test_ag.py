# Arquivo: test_ag.py
import pytest
import numpy as np

# IMPORTANTE: Aqui importamos as funções do arquivo modularizado
from ag_functions import create_individual, crossover, mutate

def test_create_individual_retorna_cromossomo_valido():
    ind = create_individual()
    assert len(ind) == 3, "O cromossomo deve ter exatamente 3 genes."
    assert isinstance(ind[0], (int, np.integer)), "n_estimators deve ser inteiro"
    assert 10 <= ind[0] <= 200, "n_estimators fora do limite"
    assert isinstance(ind[1], (int, np.integer)), "max_depth deve ser inteiro"
    assert 1 <= ind[1] <= 20, "max_depth fora do limite"
    assert isinstance(ind[2], (float, np.floating)), "min_samples_split deve ser float"
    assert 0.01 <= ind[2] <= 0.5, "min_samples_split fora do limite"

def test_crossover_mistura_genes_dos_pais():
    p1 = [100, 5, 0.1]
    p2 = [150, 15, 0.4]
    filho = crossover(p1, p2)
    assert len(filho) == 3, "O filho deve ter 3 genes"
    assert filho[0] in (p1[0], p2[0]), "Gene 0 não veio de nenhum dos pais"
    assert filho[1] in (p1[1], p2[1]), "Gene 1 não veio de nenhum dos pais"

def test_mutacao_mantem_integridade_do_cromossomo():
    ind = [100, 5, 0.1]
    # Forçamos prob=1.0 para garantir que a mutação vai ocorrer no teste
    ind_mutado = mutate(ind.copy(), prob=1.0)
    assert len(ind_mutado) == 3, "A mutação alterou o tamanho do cromossomo"
    assert isinstance(ind_mutado[0], (int, np.integer))
    assert isinstance(ind_mutado[1], (int, np.integer))
    assert isinstance(ind_mutado[2], (float, np.floating))