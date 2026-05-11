from typing import Tuple, Literal

State = Tuple[int, int, int]
Action = Literal["roll", "hold"]

def optimal_pig_value_iteration(goal: int = 100,
    epsilon: float = 1e-10,
    max_sweeps_per_partition: int = 10_000,
) -> tuple[dict[State, Action], dict[State, float]]:
    """
    Computes the optimal Pig policy using value iteration

    State is (i, j, k):
    i = current player's score
    j = opponent's score
    k = current turn total

    The value function stores the probability that the current player wins
    """
    value_func: dict[State, float] = {
        (i, j, k): 0.0
        for i in range(goal)
        for j in range(goal)
        for k in range(goal - i)
    }

    def V(i: int, j: int, k: int) -> float:
        """Return the current value estimate for a state"""
        if i + k >= goal:
            return 1.0
        return value_func[(i, j, k)]

    def q_roll(i: int, j: int, k: int) -> float:
        """Compute the value of choosing the roll action"""
        roll_one_value = 1.0 - V(j, i, 0)
        successful_roll_values = sum(V(i, j, k + die) for die in range(2, 7))
        return (roll_one_value + successful_roll_values) / 6.0

    def q_hold(i: int, j: int, k: int) -> float:
        """Compute the value of choosing the hold action"""
        if i + k >= goal:
            return 1.0
        return 1.0 - V(j, i + k, 0)

    total_sweeps = 0

    # Iterate over all possible sums of the two players' banked scores
    for score_sum in range(2 * goal - 2, -1, -1):
        # Track score pairs already handled for this total score.
        seen_score_pairs: set[frozenset[tuple[int, int]]] = set()
        # Compute the valid range of the current player's score i, given that i + j = score_sum and both scores must be below goal
        min_i = max(0, score_sum - (goal - 1))
        max_i = min(goal - 1, score_sum)
        for i in range(min_i, max_i + 1):
            j = score_sum - i
            # Skip invalid opponent scores
            if not (0 <= j < goal):
                continue
            # Treat (i, j) and (j, i) as the same partition
            pair_key = frozenset({(i, j), (j, i)})
            # If this symmetric score pair has already been processed, skip it
            if pair_key in seen_score_pairs:
                continue
            seen_score_pairs.add(pair_key)
            # Build all states in this partition
            states: list[State] = [
                (i, j, k)
                for k in range(goal - i)
            ]
            # If the scores differ, include the swapped player perspective too
            if i != j:
                states.extend(
                    (j, i, k)
                    for k in range(goal - j)
                )
            # Repeatedly sweep over this partition until its values converge or until the maximum allowed number of sweeps is reached
            for _ in range(max_sweeps_per_partition):
                # Snapshot the partition's values at the start of the sweep
                # During this sweep, all references to states inside the same partition use these old values
                old_values = {state: value_func[state] for state in states}
                def V_local(a: int, b: int, c: int) -> float:
                    """
                    Return the value of state (a, b, c).
                    If the current player has reached the goal with banked score plus turn total, the value is 1.0 because the current player
                    has already won. If the requested state is inside the current partition, use the old value from the beginning of this sweep.
                    Otherwise, use the latest globally stored value.
                    """
                    if a + c >= goal:
                        return 1.0
                    state = (a, b, c)
                    if state in old_values:
                        return old_values[state]
                    return value_func[state]
                # Stores updated values for this sweep
                new_values: dict[State, float] = {}
                # Track the largest value change during this sweep. Once this is smaller than epsilon, the partition is considered converged
                delta = 0.0
                for a, b, c in states:
                    # Expected value of rolling
                    roll_value = ((1.0 - V_local(b, a, 0)) + sum(V_local(a, b, c + die) for die in range(2, 7))) / 6.0
                    # Expected value of holding
                    hold_value = 1.0 - V_local(b, a + c, 0)
                    # The optimal value is whichever action gives the higher probability of eventually winning
                    new_value = max(roll_value, hold_value)
                    # Save the updated value for this state
                    new_values[(a, b, c)] = new_value
                    # Update the convergence metric
                    delta = max(delta, abs(new_value - old_values[(a, b, c)]))
                # Commit all updates from this sweep at once
                value_func.update(new_values)
                total_sweeps += 1
                # Stop sweeping this partition once changes are negligible
                if delta < epsilon:
                    break
            else:
                raise RuntimeError(
                    f"Partition with score sum {score_sum} did not converge"
                )
            
    # After value iteration finishes, construct the optimal policy: for each state, choose the action that achieves the state's value
    policy: dict[State, Action] = {}
    for state in value_func:
        i, j, k = state
        roll_value = q_roll(i, j, k)
        hold_value = q_hold(i, j, k)
        policy[state] = "roll" if roll_value >= hold_value else "hold"
    print(f"Total partition sweeps = {total_sweeps}")
    print(f"Starting-player win probability = {value_func[(0, 0, 0)]:.6f}")

    return policy, value_func
