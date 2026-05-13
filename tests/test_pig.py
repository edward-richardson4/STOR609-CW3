import pytest
from pytest import approx


from implementation.pig_value_iteration import optimal_pig_value_iteration

@pytest.fixture(scope="module")
def optimal_pig_data():
    """
    Computes the optimal policy and value function once for all tests.
    Uses a slightly relaxed epsilon to keep test suite execution fast.
    """
    policy, value_func = optimal_pig_value_iteration(goal=100, epsilon=1e-5)
    return policy, value_func

def test_optimal_pig_starting_probability(optimal_pig_data):
    """
    The Neller & Presser paper explicitly states that an optimal player 
    going first wins 53.06% of the time.
    """
    _, value_func = optimal_pig_data
    
    # Check the probability of winning from a 0-0 score with 0 turn total
    assert value_func[(0, 0, 0)] == approx(0.5306, abs=1e-4)

def test_state_space_size(optimal_pig_data):
    """
    The authors note that Pig presents exactly 505,000 equations.
    This ensures the state space bounds were constructed correctly.
    """
    policy, value_func = optimal_pig_data
    
    assert len(value_func) == 505_000
    assert len(policy) == 505_000

def test_initial_action_is_roll(optimal_pig_data):
    """
    At the very beginning of the game, holding is never optimal.
    """
    policy, _ = optimal_pig_data
    assert policy[(0, 0, 0)] == "roll"

def test_policy_avoids_redundant_states(optimal_pig_data):
    """
    Ensures that states where a player has already won (i + k >= 100) 
    are correctly excluded from the state space.
    """
    policy, _ = optimal_pig_data
    
    assert (80, 50, 20) not in policy
    assert (99, 50, 1) not in policy
    assert (0, 0, 100) not in policy
