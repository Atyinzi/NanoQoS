from benchmarks.run import make_workload, workload_sha256


def test_workload_is_deterministic_and_prefix_is_shared():
    left = make_workload(5, 7, "poisson", 10, 8, 16, 4)
    right = make_workload(5, 7, "poisson", 10, 8, 16, 4)
    assert left == right
    assert [item.arrival_offset for item in left] == sorted(item.arrival_offset for item in left)
    assert all(item.prompt[:8] == left[0].prompt[:8] for item in left)
    assert workload_sha256(left) == workload_sha256(right)
    assert workload_sha256(left) != workload_sha256(
        make_workload(5, 8, "poisson", 10, 8, 16, 4))
