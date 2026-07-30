from pos import pos


def test_positions_and_roles_are_cached_and_deterministic():
    _, _, first_positions, first_roles = pos()
    _, _, second_positions, second_roles = pos(seed=999)

    assert first_positions == second_positions
    assert first_roles == second_roles
    assert set(first_roles) <= {"T", "R"}
