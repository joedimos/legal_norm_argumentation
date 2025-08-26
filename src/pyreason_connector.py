from pyreason import PyReasoner, Fact, Rule, Relation
from argumentation import LegalArgument, Attack
import logging

logger = logging.getLogger(__name__)

class ArgumentationReasoner:
    def __init__(self):
        self.reasoner = PyReasoner()
        self.arguments = {}
        self.attacks = []

    def add_argument(self, argument: LegalArgument):
        """Register argument as a fact in PyReason"""
        self.arguments[argument.id] = argument
        fact = Fact(f"arg_{argument.id}", {"claim": argument.claim})
        self.reasoner.add_fact(fact)
        logger.info(f"Added argument {argument.id} to PyReason.")

    def add_attack(self, attack: Attack):
        """Register attack as a relation in PyReason"""
        self.attacks.append(attack)
        relation = Relation(
            f"attack_{attack.attacker.id}_{attack.target.id}",
            f"arg_{attack.attacker.id}",
            f"arg_{attack.target.id}"
        )
        self.reasoner.add_relation(relation)
        logger.info(f"Added attack {attack.attacker.id} -> {attack.target.id}.")

    def run_reasoning(self):
        """Execute reasoning cycle"""
        logger.info("Running PyReason...")
        self.reasoner.run()
        results = self.reasoner.get_results()
        return results
