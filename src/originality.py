from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import logging

import networkx as nx
import pyreason as pr


from .argumentation import LegalArgument, Sequent
from .norm_extraction import extract_norms
from .models import nlp

logger = logging.getLogger(__name__)

# --- Keywords ---
ORIGINALITY_KEYWORDS = {
    "empreinte de la personnalité", "author's personality",
    "choix créatifs libres", "free creative choices",
    "dictated by technical"
}


class PyReasonBridge:
    """
    Bridge layer between our LegalArgument/Sequent representation
    and the PyReason reasoning engine.
    
    Uses the correct PyReason API with graph construction, rules, and facts.
    """

    def __init__(self):
        self.graph = pr.Graph()
        self.arguments: Dict[str, LegalArgument] = {}
        self.facts_added: Set[str] = set()
        self.rules_added: Set[str] = set()
        self.initialized = False

    def initialize(self) -> None:
        """Initialize PyReason environment"""
        if not self.initialized:
            pr.reset()
            pr.reset_rules()
            self.initialized = True
            logger.info("PyReason environment initialized")

    def add_argument_as_node(self, arg: LegalArgument) -> None:
        """
        Add a LegalArgument as a node in the PyReason graph.
        """
        if not self.initialized:
            self.initialize()
            
        # Add node to graph
        self.graph.add_node(arg.id)
        self.arguments[arg.id] = arg
        logger.debug(f"Added argument node: {arg.id}")

    def add_attack_edge(self, attacker_id: str, target_id: str) -> None:
        """Add an attack relationship as an edge"""
        if attacker_id in self.arguments and target_id in self.arguments:
            self.graph.add_edge(attacker_id, target_id, label='attacks')
            logger.debug(f"Added attack edge: {attacker_id} -> {target_id}")

    def add_support_edge(self, supporter_id: str, supported_id: str) -> None:
        """Add a support relationship as an edge"""
        if supporter_id in self.arguments and supported_id in self.arguments:
            self.graph.add_edge(supporter_id, supported_id, label='supports')
            logger.debug(f"Added support edge: {supporter_id} -> {supported_id}")

    def setup_rules(self) -> None:
        """
        Define PyReason rules for legal argumentation.
        """
        # Rule 1: An argument is defeated if attacked by an accepted argument
        pr.add_rule(pr.Rule(
            'defeated(Y) <- attacks(X, Y), accepted(X)',
            'argument_defeated',
            infer_edges=True
        ))
        self.rules_added.add('argument_defeated')
        
        # Rule 2: An argument is reinstated if its attacker is defeated
        pr.add_rule(pr.Rule(
            'reinstated(Y) <- attacks(X, Y), defeated(X)',
            'argument_reinstated',
            infer_edges=True
        ))
        self.rules_added.add('argument_reinstated')
        
        # Rule 3: Supported arguments are strengthened
        pr.add_rule(pr.Rule(
            'strengthened(Y) <- supports(X, Y), accepted(X)',
            'argument_strengthened',
            infer_edges=True
        ))
        self.rules_added.add('argument_strengthened')
        
        # Rule 4: Original works get copyright protection
        pr.add_rule(pr.Rule(
            'protected(X) <- original(X), accepted(X)',
            'originality_implies_protection',
            infer_edges=False
        ))
        self.rules_added.add('originality_implies_protection')
        
        # Rule 5: Technically dictated works are not original
        pr.add_rule(pr.Rule(
            'not_original(X) <- technical_function(X), accepted(X)',
            'technical_not_original',
            infer_edges=False
        ))
        self.rules_added.add('technical_not_original')
        
        logger.info(f"Added {len(self.rules_added)} PyReason rules")

    def set_initial_facts(self) -> None:
        """
        Set initial facts for arguments based on their confidence and type.
        """
        for arg_id, arg in self.arguments.items():
            # Convert confidence tuple to bounds
            if isinstance(arg.confidence, tuple):
                lower, upper = arg.confidence
            else:
                lower = upper = arg.confidence
            
            # Set acceptance based on confidence
            pr.add_fact(pr.Fact(
                arg_id,
                'accepted',
                [lower, upper],
                timestep=0
            ))
            self.facts_added.add(f"{arg_id}_accepted")
            
            # Add domain-specific predicates based on argument content
            conclusion_lower = arg.conclusion.lower()
            
            if 'original' in conclusion_lower and 'not' not in conclusion_lower:
                pr.add_fact(pr.Fact(
                    arg_id,
                    'original',
                    [lower * 0.9, upper],
                    timestep=0
                ))
                self.facts_added.add(f"{arg_id}_original")
            
            if 'technical' in arg.premise.lower() or 'function' in arg.premise.lower():
                pr.add_fact(pr.Fact(
                    arg_id,
                    'technical_function',
                    [lower, upper],
                    timestep=0
                ))
                self.facts_added.add(f"{arg_id}_technical")
            
            if 'personality' in arg.premise.lower() or 'creative' in arg.premise.lower():
                pr.add_fact(pr.Fact(
                    arg_id,
                    'creative',
                    [lower, upper],
                    timestep=0
                ))
                self.facts_added.add(f"{arg_id}_creative")
        
        logger.info(f"Set {len(self.facts_added)} initial facts")

    def run_reasoning(self, timesteps: int = 10) -> Dict:
        """
        Execute PyReason reasoning and return results.
        
        Parameters
        ----------
        timesteps : int
            Number of reasoning timesteps
            
        Returns
        -------
        Dict
            Reasoning results with acceptance, defeat, and protection scores
        """
        try:
            # Load graph
            pr.load_graph(self.graph)
            logger.info(f"Loaded graph with {len(self.arguments)} nodes")
            
            # Setup rules and facts
            self.setup_rules()
            self.set_initial_facts()
            
            # Run reasoning
            logger.info(f"Running PyReason for {timesteps} timesteps...")
            interpretation = pr.reason(timesteps=timesteps)
            
            # Extract results
            results = self._extract_results(interpretation, timesteps)
            
            logger.info("PyReason reasoning completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"PyReason reasoning failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _extract_results(self, interpretation: Dict, timesteps: int) -> Dict:
        """
        Extract and format reasoning results from PyReason interpretation.
        """
        results = {
            "accepted": {},
            "defeated": {},
            "reinstated": {},
            "original": {},
            "protected": {},
            "convergence": False
        }
        
        try:
            # Get final timestep
            final_timestep = min(timesteps - 1, max(interpretation.keys()))
            
            for arg_id in self.arguments.keys():
                if arg_id in interpretation[final_timestep]:
                    node_data = interpretation[final_timestep][arg_id]
                    
                    # Extract predicates with bounds
                    if 'accepted' in node_data:
                        bounds = node_data['accepted']
                        results['accepted'][arg_id] = {
                            'lower': bounds[0],
                            'upper': bounds[1],
                            'mean': (bounds[0] + bounds[1]) / 2
                        }
                    
                    if 'defeated' in node_data:
                        bounds = node_data['defeated']
                        results['defeated'][arg_id] = {
                            'lower': bounds[0],
                            'upper': bounds[1],
                            'mean': (bounds[0] + bounds[1]) / 2
                        }
                    
                    if 'reinstated' in node_data:
                        bounds = node_data['reinstated']
                        results['reinstated'][arg_id] = {
                            'lower': bounds[0],
                            'upper': bounds[1],
                            'mean': (bounds[0] + bounds[1]) / 2
                        }
                    
                    if 'original' in node_data:
                        bounds = node_data['original']
                        results['original'][arg_id] = {
                            'lower': bounds[0],
                            'upper': bounds[1],
                            'mean': (bounds[0] + bounds[1]) / 2
                        }
                    
                    if 'protected' in node_data:
                        bounds = node_data['protected']
                        results['protected'][arg_id] = {
                            'lower': bounds[0],
                            'upper': bounds[1],
                            'mean': (bounds[0] + bounds[1]) / 2
                        }
            
            # Check convergence
            if len(interpretation) >= 2:
                results['convergence'] = self._check_convergence(interpretation)
            
        except Exception as e:
            logger.error(f"Error extracting results: {e}", exc_info=True)
        
        return results

    def _check_convergence(self, interpretation: Dict) -> bool:
        """Check if reasoning has converged by comparing last two timesteps"""
        timesteps = sorted(interpretation.keys())
        if len(timesteps) < 2:
            return False
        
        last = interpretation[timesteps[-1]]
        second_last = interpretation[timesteps[-2]]
        
        # Compare node states
        return last == second_last

    def query_acceptance(self, arg_id: str, results: Dict) -> float:
        """
        Query the acceptance score of an argument from results.
        
        Parameters
        ----------
        arg_id : str
            Argument identifier
        results : Dict
            Results from run_reasoning()
            
        Returns
        -------
        float
            Acceptance score (0.0 to 1.0)
        """
        if arg_id in results.get('accepted', {}):
            return results['accepted'][arg_id]['mean']
        return 0.0

    def query_originality(self, results: Dict) -> float:
        """
        Aggregate originality score across all original arguments.
        """
        original_scores = [
            data['mean'] 
            for data in results.get('original', {}).values()
        ]
        if not original_scores:
            return 0.0
        return sum(original_scores) / len(original_scores)

    def query_protection(self, results: Dict) -> float:
        """
        Aggregate copyright protection score.
        """
        protection_scores = [
            data['mean']
            for data in results.get('protected', {}).values()
        ]
        if not protection_scores:
            return 0.0
        return sum(protection_scores) / len(protection_scores)


# -------------------------
# OriginalityAnnotator
# -------------------------
class OriginalityAnnotator:
    """
    Extracts originality-related claims from text, builds legal arguments,
    constructs sequents, and attaches them to the reasoner.
    """

    def __init__(self, reasoner: "LogicBasedLegalReasoner"):
        self.reasoner = reasoner
        self._current_arguments: Dict[str, LegalArgument] = {}
        self._current_attacks: Set[Tuple[str, str]] = set()
        self._current_supports: Set[Tuple[str, str]] = set()
        self.sequents: List[Sequent] = []

    def extract_originality_claims(self, text: str) -> List[Dict]:
        """Extract originality-related claims from text"""
        claims = []
        t = text.lower()

        if any(kw in t for kw in ORIGINALITY_KEYWORDS):
            if "empreinte de la personnalité" in t or "author's personality" in t:
                claims.append({
                    "premise": "Work shows the imprint of the author's personality.",
                    "conclusion": "The work is original.",
                    "confidence": 0.88,
                })
            if "choix créatifs libres" in t or "free creative choices" in t:
                claims.append({
                    "premise": "Author made free and creative choices.",
                    "conclusion": "The work is original.",
                    "confidence": 0.91,
                })
        if "dictated by technical function" in t:
            claims.extend([
                {
                    "premise": "Form is dictated by technical function.",
                    "conclusion": "The work is NOT original.",
                    "confidence": 0.95,
                },
                {
                    "premise": "Functional elements lack originality.",
                    "conclusion": "No copyright protection for functional aspects.",
                    "confidence": 0.90,
                },
            ])
        return claims

    def _create_arguments_from_claims(self, claims: List[Dict]) -> Dict[str, LegalArgument]:
        """Create LegalArgument objects from extracted claims"""
        arguments: Dict[str, LegalArgument] = {}
        for i, claim in enumerate(claims):
            conf = claim.get("confidence", 0.8)
            low, high = conf * 0.8, conf

            supports_originality = "not" not in claim["conclusion"].lower()
            if supports_originality:
                orig_id = f"is_original_{i}"
                arguments[orig_id] = LegalArgument(
                    id=orig_id,
                    premise=claim["premise"],
                    conclusion=claim["conclusion"],
                    confidence=(low, high),
                    supporting_args=frozenset(),
                    attacking_args=frozenset(),
                    sequent_type="hypothesis",
                )
                prot_id = f"copyright_protected_{i}"
                arguments[prot_id] = LegalArgument(
                    id=prot_id,
                    premise=f"Based on {orig_id}",
                    conclusion="Copyright protection likely applies.",
                    confidence=(low * 0.9, high),
                    supporting_args=frozenset([orig_id]),
                    attacking_args=frozenset(),
                    sequent_type="conclusion",
                )
                # Track support relationship
                self._current_supports.add((orig_id, prot_id))
            else:
                not_id = f"not_original_{i}"
                arguments[not_id] = LegalArgument(
                    id=not_id,
                    premise=claim["premise"],
                    conclusion=claim["conclusion"],
                    confidence=(low, high),
                    supporting_args=frozenset(),
                    attacking_args=frozenset(),
                    sequent_type="hypothesis",
                )
                notprot_id = f"copyright_not_protected_{i}"
                arguments[notprot_id] = LegalArgument(
                    id=notprot_id,
                    premise=f"Based on {not_id}",
                    conclusion="Copyright protection likely does not apply.",
                    confidence=(low * 0.9, high),
                    supporting_args=frozenset([not_id]),
                    attacking_args=frozenset(),
                    sequent_type="conclusion",
                )
                # Track support relationship
                self._current_supports.add((not_id, notprot_id))

        self._link_attacks(arguments)
        return arguments

    def _link_attacks(self, arguments: Dict[str, LegalArgument]) -> None:
        """Identify and link attack relationships between arguments"""
        ids = list(arguments.keys())
        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1:]:
                a, b = arguments[a_id], arguments[b_id]
                if (
                    ("original" in a.conclusion.lower() and "not original" in b.conclusion.lower())
                    or ("original" in b.conclusion.lower() and "not original" in a.conclusion.lower())
                ):
                    self._current_attacks.add((a.id, b.id))
                    self._current_attacks.add((b.id, a.id))

    def construct_sequents(self) -> None:
        """Construct sequents from current arguments"""
        args = self._current_arguments
        originality_args = {k for k in args if "is_original" in k}
        protection_args = {k for k in args if "copyright_protected" in k}
        no_protection_args = {k for k in args if "copyright_not_protected" in k}
        not_original_args = {k for k in args if "not_original" in k}

        sequents = []
        for orig in originality_args:
            conf = args[orig].confidence[0]
            sequents.append(
                Sequent(
                    antecedents=frozenset([orig]),
                    consequents=frozenset(protection_args),
                    confidence=conf * 0.9,
                )
            )
            sequents.append(
                Sequent(
                    antecedents=frozenset(no_protection_args),
                    consequents=frozenset(not_original_args),
                    confidence=conf * 0.9,
                    is_contrapositive=True,
                )
            )
        self.sequents = sequents

    def annotate_originality(self, text: str) -> Dict:
        """Main orchestration method for originality annotation"""
        self._current_arguments.clear()
        self._current_attacks.clear()
        self._current_supports.clear()
        self.sequents = []

        claims = self.extract_originality_claims(text)
        if not claims:
            return {
                "arguments": {},
                "attacks": set(),
                "supports": set(),
                "sequents": [],
                "norms": []
            }

        self._current_arguments = self._create_arguments_from_claims(claims)
        self.construct_sequents()

        # Add to reasoner
        for arg in self._current_arguments.values():
            self.reasoner.add_argument(arg)

        norms = extract_norms(text, use_transformer=False)

        return {
            "arguments": self._current_arguments,
            "attacks": self._current_attacks,
            "supports": self._current_supports,
            "sequents": self.sequents,
            "norms": norms,
        }


# -------------------------
# LogicBasedLegalReasoner
# -------------------------
class LogicBasedLegalReasoner:
    """
    Coordinates argument construction, applies IP-specific rules,
    and performs reasoning via PyReason.
    """

    def __init__(self):
        self.arguments: Dict[str, LegalArgument] = {}
        self.annotator = OriginalityAnnotator(self)
        self.pyreason = PyReasonBridge()
        self.attacks: Set[Tuple[str, str]] = set()
        self.supports: Set[Tuple[str, str]] = set()

    def add_argument(self, arg: LegalArgument) -> None:
        """Add an argument to the reasoner"""
        self.arguments[arg.id] = arg
        self.pyreason.add_argument_as_node(arg)

    def add_attack(self, attacker_id: str, target_id: str) -> None:
        """Add an attack relationship"""
        self.attacks.add((attacker_id, target_id))
        self.pyreason.add_attack_edge(attacker_id, target_id)

    def add_support(self, supporter_id: str, supported_id: str) -> None:
        """Add a support relationship"""
        self.supports.add((supporter_id, supported_id))
        self.pyreason.add_support_edge(supporter_id, supported_id)

    def analyze_legal_case(self, text: str, timesteps: int = 10) -> Dict:
        """
        Main analysis method for legal cases.
        
        Parameters
        ----------
        text : str
            Legal case text to analyze
        timesteps : int
            Number of PyReason reasoning timesteps
            
        Returns
        -------
        Dict
            Complete analysis results
        """
        # Extract and annotate
        result = self.annotator.annotate_originality(text)

        # Add relationships to PyReason
        for attacker, target in result["attacks"]:
            self.add_attack(attacker, target)
        
        for supporter, supported in result["supports"]:
            self.add_support(supporter, supported)

        # Run PyReason reasoning
        pyreason_results = self.pyreason.run_reasoning(timesteps=timesteps)

        # Query scores
        originality_score = self.pyreason.query_originality(pyreason_results)
        protection_score = self.pyreason.query_protection(pyreason_results)

        return {
            "arguments": {k: v.__dict__ for k, v in result["arguments"].items()},
            "attacks": list(result["attacks"]),
            "supports": list(result["supports"]),
            "sequents": [
                {
                    "antecedents": list(s.antecedents),
                    "consequents": list(s.consequents),
                    "confidence": s.confidence,
                    "is_contrapositive": s.is_contrapositive
                }
                for s in result["sequents"]
            ],
            "norms": result["norms"],
            "pyreason_results": pyreason_results,
            "pyreason_eval": {
                "originality_score": originality_score,
                "protection_score": protection_score,
            },
        }
