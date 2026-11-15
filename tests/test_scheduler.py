import pytest

from chunk_trajectory_error.scheduler import ChunkScheduler


def run(horizon, execute, delay, steps):
    scheduler = ChunkScheduler(horizon, execute, delay)
    records, calls = [], []
    for k in range(steps):
        def predict(k=k):
            calls.append(k)
            return [(k, index) for index in range(horizon)]  # command encodes (observation step, chunk index)
        command, record = scheduler.step(k, predict)
        assert command == (record.observed_at, record.chunk_index)
        records.append(record)
    return records, calls


@pytest.mark.parametrize("delay", [0, 2, 4])
def test_every_step_gets_exactly_one_command_and_segments_are_contiguous(delay):
    execute, steps = 5, 37
    records, _ = run(20, execute, delay, steps)
    assert [r.step for r in records] == list(range(steps))  # no gaps, no repeats
    assert [r.chunk_index for r in records[:execute]] == list(range(execute))  # first chunk is synchronous
    for record in records[execute:]:
        segment_step = (record.step - execute) % execute
        assert record.chunk_index == delay + segment_step  # new chunk executes A[r : r + e]
        assert record.observed_at == record.step - segment_step - delay  # predicted from the observation at b - r
        assert record.is_boundary == (segment_step == 0)


def test_readme_example_e10_r4():
    records, calls = run(50, 10, 4, 25)
    assert calls[:3] == [0, 6, 16]  # request at the 6-step mark, takeover 4 steps later
    assert [r.chunk_index for r in records[10:20]] == list(range(4, 14))
    assert records[10].is_boundary and records[10].observed_at == 6
    assert all(r.chunk_id == 0 for r in records[6:10])  # the pending result never affects the old segment


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        ChunkScheduler(8, 5, 4)
    with pytest.raises(ValueError):
        ChunkScheduler(20, 5, 6)
