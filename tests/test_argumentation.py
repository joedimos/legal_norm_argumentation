from src.argumentation import OptimizedAAF
def test_aaf_basic():
    args = {"a", "b"}
    attacks = {("a","b")}
    aaf = OptimizedAAF(args, attacks)
    assert aaf.is_conflict_free({"a"})
    assert not aaf.is_conflict_free({"a","b"})
