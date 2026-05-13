import pytest
from pytest import approx

from code.piglet import piglet_value_iteration

@pytest.fixture(scope="module")
def piglet_final_values():
    """
    Runs the Piglet value iteration once for the entire test module.
    30 iterations is enough to converge to the exact fractions.
    """
    _, final_values = piglet_value_iteration(iterations=30)
    return final_values

@pytest.mark.parametrize("state, expected_probability", [
    ((0, 0, 0), 4/7),
    ((0, 0, 1), 5/7),
    ((0, 1, 0), 2/5),
    ((0, 1, 1), 3/5),
    ((1, 0, 0), 4/5),
    ((1, 1, 0), 2/3),
])
def test_piglet_exact_probabilities(piglet_final_values, state, expected_probability):
    """
    Verifies that the simulated probabilities match the exact fractions 
    from the Neller & Presser paper.
    """
    assert piglet_final_values[state] == approx(expected_probability, abs=1e-4)

def test_piglet_winning_state_behavior(piglet_final_values):
    """
    Verifies that the state space only tracks non-winning states.
    States where i + k >= 2 should not be in the dictionary.
    """
    assert (0, 0, 2) not in piglet_final_values
    assert (1, 0, 1) not in piglet_final_values
