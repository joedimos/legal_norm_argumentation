from __future__ import annotations

from typing import Dict, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class PyReasonConnector:
    """Optional adapter around PyReason.

    The verified core does not depend on this module. The adapter fails closed
    with a structured result when PyReason is unavailable or incompatible.
    """

    @staticmethod
    def _load_pyreason():
        try:
            import pyreason as pr
            return pr
        except Exception as exc:
            raise RuntimeError(
                "PyReason is not installed or could not be imported. "
                "Install the optional requirements in a compatible Python environment."
            ) from exc

    def run_reasoning(
        self,
        arguments: Dict[str, Dict],
        attacks: Set[Tuple[str, str]],
        supports: Optional[Set[Tuple[str, str]]] = None,
        timesteps: int = 10,
    ) -> Dict:
        if not arguments:
            return {"accepted": {}, "defeated": {}, "reinstated": {}, "strengthened": {}, "converged": True}

        try:
            pr = self._load_pyreason()
            import networkx as nx

            pr.reset()
            if hasattr(pr, "reset_rules"):
                pr.reset_rules()

            graph = nx.DiGraph()
            graph.add_nodes_from(arguments)
            for edge in attacks | (supports or set()):
                graph.add_edge(*edge)
            pr.load_graph(graph)

            end_time = max(0, timesteps - 1)
            for arg_id, metadata in arguments.items():
                confidence = metadata.get("confidence", (0.8, 1.0))
                low, high = confidence if isinstance(confidence, (tuple, list)) else (confidence, confidence)
                fact_text = f"accepted({arg_id}):[{float(low)},{float(high)}]"
                pr.add_fact(pr.Fact(fact_text, "input_acceptance", 0, end_time))

            for source, target in attacks:
                pr.add_fact(pr.Fact(f"attacks({source},{target}):[1,1]", "attack", 0, end_time))
            for source, target in supports or set():
                pr.add_fact(pr.Fact(f"supports({source},{target}):[1,1]", "support", 0, end_time))

            pr.add_rule(pr.Rule("defeated(Y):[1,1] <- attacks(X,Y):[1,1], accepted(X):[0.5,1]", "defeat_rule"))
            pr.add_rule(pr.Rule("reinstated(Y):[1,1] <- attacks(X,Y):[1,1], defeated(X):[1,1]", "reinstate_rule"))
            if supports:
                pr.add_rule(pr.Rule("strengthened(Y):[1,1] <- supports(X,Y):[1,1], accepted(X):[0.5,1]", "support_rule"))

            interpretation = pr.reason(timesteps=timesteps)
            return self._extract_results(interpretation, set(arguments))
        except Exception as exc:
            logger.warning("PyReason backend unavailable/incompatible: %s", exc)
            return {
                "error": "pyreason_unavailable_or_incompatible",
                "details": str(exc),
                "accepted": {},
                "defeated": {},
                "reinstated": {},
                "strengthened": {},
                "converged": False,
            }

    @staticmethod
    def _extract_results(interpretation, arg_ids: Set[str]) -> Dict:
        result = {
            "accepted": {},
            "defeated": {},
            "reinstated": {},
            "strengthened": {},
            "converged": False,
        }
        if interpretation is None:
            return result

        data = interpretation.get_dict() if hasattr(interpretation, "get_dict") else interpretation
        if not isinstance(data, dict) or not data:
            return result
        times = sorted(data)
        final = data[times[-1]]

        for arg_id in arg_ids:
            node_data = final.get(arg_id, {}) if isinstance(final, dict) else {}
            for predicate in ("accepted", "defeated", "reinstated", "strengthened"):
                value = node_data.get(predicate) if isinstance(node_data, dict) else None
                if value is not None:
                    result[predicate][arg_id] = value
        if len(times) > 1:
            result["converged"] = data[times[-1]] == data[times[-2]]
        return result
