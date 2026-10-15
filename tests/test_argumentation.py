from src.argumentation import LegalArgument, OptimizedAAF


def test_aaf_basic_and_empty_admissible():
    aaf = OptimizedAAF({"a", "b"}, {("a", "b")})
    assert aaf.is_conflict_free({"a"})
    assert not aaf.is_conflict_free({"a", "b"})
    assert set() in aaf.get_admissible_sets()
    assert aaf.get_grounded_extension() == {"a"}
    assert aaf.grounded_labeling() == {"a": "in", "b": "out"}


def test_self_attack_is_not_conflict_free():
    aaf = OptimizedAAF({"a"}, {("a", "a")})
    assert not aaf.is_conflict_free({"a"})
    assert aaf.get_preferred_extensions() == [set()]
    assert aaf.get_grounded_extension() == set()


def test_reinstatement_chain_grounded():
    aaf = OptimizedAAF({"a", "b", "c"}, {("a", "b"), ("b", "c")})
    assert aaf.get_grounded_extension() == {"a", "c"}
    assert aaf.grounded_labeling() == {"a": "in", "b": "out", "c": "in"}


def test_support_does_not_pollute_attack_semantics():
    aaf = OptimizedAAF({"a", "b"}, attacks=set(), supports={("a", "b")})
    assert aaf.is_conflict_free({"a", "b"})
    assert aaf.supports == {("a", "b")}
    assert aaf.attacks == set()


def test_build_argument_graph_preserves_relations():
    args = [
        LegalArgument("a", attacking_args=frozenset({"b"})),
        LegalArgument("b"),
        LegalArgument("c", supporting_args=frozenset({"b"})),
    ]
    graph = OptimizedAAF.build_argument_graph(args)
    relations = {(u, v, data["relation"]) for u, v, data in graph.edges(data=True)}
    assert ("a", "b", "attacks") in relations
    assert ("c", "b", "supports") in relations


def test_all_three_argument_attack_graphs_satisfy_semantic_invariants():
    ids = ("a", "b", "c")
    possible_edges = [(a, b) for a in ids for b in ids]
    for mask in range(1 << len(possible_edges)):
        attacks = {edge for index, edge in enumerate(possible_edges) if mask & (1 << index)}
        aaf = OptimizedAAF(ids, attacks=attacks)
        grounded = aaf.get_grounded_extension()
        preferred = aaf.get_preferred_extensions()
        stable = aaf.get_stable_extensions()

        assert aaf.is_conflict_free(grounded)
        assert all(aaf.defends(grounded, arg) for arg in grounded)
        assert all(aaf.is_conflict_free(ext) for ext in preferred)
        assert all(all(aaf.defends(ext, arg) for arg in ext) for ext in preferred)
        assert all(aaf.is_conflict_free(ext) for ext in stable)
        for ext in stable:
            assert set(ids) - ext <= {target for source in ext for target in aaf.attacks_dict.get(source, set())}
