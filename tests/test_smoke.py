from ml_lab.app import restart_recovery_smoke_test, smoke_test


def test_non_gui_smoke() -> None:
    assert smoke_test() == 0


def test_restart_recovery_smoke() -> None:
    assert restart_recovery_smoke_test() == 0
