from pos import pos, r_data


def test_positions_and_roles_are_cached_and_deterministic():
    _, _, first_positions, first_roles = pos()
    _, _, second_positions, second_roles = pos(seed=999)

    assert first_positions == second_positions
    assert first_roles == second_roles
    assert set(first_roles) <= {"T", "R"}


def test_r_data_uses_one_random_partner_per_transmitter():
    node_positions = [(0, 0), (1, 1), (2, 2), (3, 3)]
    roles = ["T", "T", "R", "R"]

    pairs, _ = r_data(node_positions, roles)

    assert len(pairs) == 2
    assert sorted(tx_idx for tx_idx, _ in pairs) == [0, 1]
    assert all(tx_idx != rx_idx for tx_idx, rx_idx in pairs)
    assert all(rx_idx in {2, 3} for _, rx_idx in pairs)
    assert len({rx_idx for _, rx_idx in pairs}) == len(pairs)
