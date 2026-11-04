"""Tests for the shot-limited optimisers."""

import torch

from cost_locality_bp.training import (Evaluator, init_theta, random_search, train_adam,
                                       train_spsa)


def test_adam_reduces_infidelity_small_system():
    gen = torch.Generator().manual_seed(0)
    n, n_layers = 3, 2
    th0 = init_theta(n, n_layers, gen)
    from cost_locality_bp.training import ideal_metrics
    start, _ = ideal_metrics(th0, n)
    res = train_adam(Evaluator(n, n_layers, 0.0, "global", 400), th0, steps=25)
    assert res.ideal_infidelity < start
    assert res.shots_used == 25 * 2 * n * n_layers * 400


def test_spsa_and_random_search_run_and_count_shots():
    gen = torch.Generator().manual_seed(1)
    n, n_layers = 3, 2
    th0 = init_theta(n, n_layers, gen)
    ev = Evaluator(n, n_layers, 0.01, "local", 100)
    r = train_spsa(ev, th0, evals=40, gen=gen)
    assert r.shots_used == 40 * 100 and 0 <= r.ideal_infidelity <= 1
    r2 = random_search(Evaluator(n, n_layers, 0.01, "global", 100), n_layers, n, 40, gen)
    assert r2.shots_used == 40 * 100
