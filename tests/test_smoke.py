from ml_lab.app import smoke_test


def test_non_gui_smoke() -> None:
    assert smoke_test() == 0
