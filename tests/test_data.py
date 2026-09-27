from knn.utils import load_groups, load_split


def test_train_subject_ids_align_with_rows() -> None:
    X, _ = load_split("train")
    groups = load_groups("train")

    assert groups is not None
    assert len(groups) == len(X)
    assert groups.nunique() == 21


def test_test_subject_ids_align_with_rows() -> None:
    X, _ = load_split("test")
    groups = load_groups("test")

    assert groups is not None
    assert len(groups) == len(X)
    assert groups.nunique() == 9


def test_no_subject_appears_in_both_splits() -> None:
    train = load_groups("train")
    test = load_groups("test")

    assert train is not None and test is not None
    assert set(train).isdisjoint(set(test))
