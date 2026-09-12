from way4.toolbox import cheapest_insertion, rolling_subset_dp, two_opt, greedy_set_cover, information_gain


def d(a, b):
    return abs(a - b)


def test_cheapest_insertion_and_two_opt_are_real_kernels():
    route, value = cheapest_insertion([10, 20], 15, 0, d)
    assert route == [10, 15, 20]
    repaired, repaired_value = two_opt([20, 10, 30], 0, d)
    assert repaired_value <= value + 30
    assert set(repaired) == {10, 20, 30}


def test_rolling_subset_dp_exact_small_route():
    route, value = rolling_subset_dp([1, 2, 3], 0, d, k=3)
    assert route[0] == 1
    assert value == 3.0


def test_greedy_cover_and_information_gain():
    assert greedy_set_cover({1, 2, 3}, [{1, 2}, {3}]) == [0, 1]
    assert information_gain([0.5, 0.5], [(1.0, [1.0, 0.0])]) > 0.0
