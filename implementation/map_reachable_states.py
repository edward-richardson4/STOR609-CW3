from __future__ import annotations
from pathlib import Path
from typing import Mapping
import numpy as np
from pig_value_iteration import Action, State, optimal_pig_value_iteration

# A policy maps each Pig game state to an action: either "roll" or "hold"
Policy = Mapping[State, Action]

def roll_die(die_sides: int = 6) -> int:
    """
    Simulate a fair die roll
    """
    return int(np.random.randint(1, die_sides + 1))

def policy_action(
    policy: Policy,
    current_score: int,
    opponent_score: int,
    turn_total: int,
    goal: int,
) -> Action:
    """
    Return the policy action for the current state.
    """
    # If holding would win the game, hold
    if current_score + turn_total >= goal:
        return "hold"
    # Otherwise, ask the optimal policy what to do in this state
    return policy[(current_score, opponent_score, turn_total)]

def game_pig(
    policy_player_1: Policy,
    policy_player_2: Policy,
    reachable: np.ndarray,
    hold_probability: float,
    *,
    goal: int = 100,
    die_sides: int = 6,
) -> np.ndarray:
    """
    Simulate one full game of Pig and update the reachable-state array.
    The reachable array records states reached by player 1. A state is stored as: reachable[player_1_score, player_2_score, turn_total] = 1
    """
    # Validate the probability used for player 2's random holding behavior
    if not 0.0 <= hold_probability <= 1.0:
        raise ValueError("hold_probability must be between 0 and 1 inclusive")
    # scores[0] is player 1's score
    # scores[1] is player 2's score
    scores = [0, 0]
    # Player 1 starts
    # player = 0 means player 1's turn
    # player = 1 means player 2's turn
    player = 0
    # Continue the game until either player reaches the goal
    while max(scores) < goal:
        # Points accumulated during the current turn
        # These are not added to the player's score until they hold
        turn_total = 0
        # Run one player's turn
        while True:
            current_score = scores[player]
            opponent_score = scores[1 - player]
            # Select the correct policy for the current player
            policy = policy_player_1 if player == 0 else policy_player_2
            # Decide whether the current player should roll or hold
            action = policy_action(
                policy,
                current_score,
                opponent_score,
                turn_total,
                goal,
            )
            if action == "roll":
                # Player 2 sometimes holds even when the policy says "roll"
                # This creates more varied games and helps reveal more of the reachable state space
                if player == 1 and np.random.random() < hold_probability:
                    scores[player] += turn_total
                    break
                # Roll the die
                roll = roll_die(die_sides)
                # Rolling a 1 ends the turn and loses the current turn total
                if roll == 1:
                    turn_total = 0
                    break
                # Any roll other than 1 is added to the turn total
                turn_total += roll
                # Record the state if it was reached during player 1's turn
                if player == 0:
                    reachable[
                        min(scores[0], goal),
                        min(scores[1], goal),
                        min(turn_total, goal),
                    ] = 1
            elif action == "hold":
                # Holding banks the turn total into the player's score
                scores[player] += turn_total
                # Record the state after player 1 holds
                # The turn total resets to 0
                if player == 0:
                    reachable[
                        min(scores[0], goal),
                        min(scores[1], goal),
                        0,
                    ] = 1
                break
        # Switch to the other player
        player = 1 - player
    return reachable

def modelling_state_space(
    policy: Policy,
    reachable: np.ndarray,
    iterations: int,
    hold_probability: float,
    *,
    goal: int = 100,
    die_sides: int = 6,
) -> np.ndarray:
    """
    Simulate many games to explore the reachable state space. Each simulated game updates the same reachable-state array.
    """
    for _ in range(iterations):
        reachable = game_pig(
            policy,
            policy,
            reachable,
            hold_probability,
            goal=goal,
            die_sides=die_sides,
        )
    return reachable

def reachable_states_where_rolling_is_optimal(
    policy: Policy,
    reachable: np.ndarray,
    *,
    goal: int = 100,
) -> np.ndarray:
    """
    Return every reachable state for which the optimal policy says to roll.

    The returned array has shape (n, 3). Each row is a state stored as:
    [current_score, opponent_score, turn_total].
    """
    rolling_states: list[State] = []
    for current_score, opponent_score, turn_total in np.argwhere(reachable == 1):
        state = (
            int(current_score),
            int(opponent_score),
            int(turn_total),
        )
        if policy_action(policy, *state, goal) == "roll":
            rolling_states.append(state)
    return np.array(rolling_states, dtype=np.int16)

def main() -> None:
    """
    Compute an optimal Pig policy, simulate many games, and save the reachable states. The simulation tries several values of
    hold_probability. Larger values make player 2 behave more randomly, which helps to explore states that may be rare or
    impossible under two perfectly optimal players.
    """
    die_sides = 6
    goal = 100
    # Number of games to simulate for each hold probability
    iterations_per_probability = 10**6
    # These values control how often player 2 randomly holds when the policy says roll
    hold_probabilities = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # Compute the optimal Pig policy using value iteration
    # The returned policy maps each state to the optimal action
    policy, _ = optimal_pig_value_iteration(
        goal=goal,
        epsilon=1e-6,
    )
    # reachable_states[player_1_score, player_2_score, turn_total] == 1 means that state was encountered during player 1's turn
    # The array has size goal + 1 in each dimension so that scores from 0 through goal can be represented
    reachable_states = np.zeros(
        (goal + 1, goal + 1, goal + 1),
        dtype=np.uint8,
    )
    # Run the simulation for each probability value
    for probability in hold_probabilities:
        reachable_states = modelling_state_space(
            policy,
            reachable_states,
            iterations=iterations_per_probability,
            hold_probability=probability,
            goal=goal,
            die_sides=die_sides,
        )
        print(f"finished looping for p = {probability}", flush=True)
    # Save the reachable-state array
    output_path = Path("reachable_states.npy")
    np.save(output_path, reachable_states)
    print(f"saved reachable states to {output_path}", flush=True)

    # Save the subset of reachable states where the optimal action is to roll.
    # This file contains an array of shape (n, 3), where each row is:
    # [current_score, opponent_score, turn_total].
    reachable_rolling_states = reachable_states_where_rolling_is_optimal(
        policy,
        reachable_states,
        goal=goal,
    )
    rolling_output_path = Path("reachable_states_roll_optimal.npy")
    np.save(rolling_output_path, reachable_rolling_states)
    print(
        f"saved {len(reachable_rolling_states)} reachable roll-optimal states to "
        f"{rolling_output_path}",
        flush=True,
    )

if __name__ == "__main__":
    main()
