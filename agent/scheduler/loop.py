"""24/7 control loop skeleton."""

import time
from agent.rl.environment import VirtualLabEnv, EnvConfig


class DummyPredictors:
    def score(self, mol):
        # placeholder: return neutral-ish scores
        return {"efficacy": 0.5, "human_safety": 0.5, "environment": 0.5, "synth_access": 0.5}


class DummyRules:
    def check(self, mol):
        return {"hard_ok": True, "soft_penalty": 0.0, "soft_violations": []}


def run_forever():
    env = VirtualLabEnv(
        cfg=EnvConfig(representation_mode="smiles", ais_vocab_path=None),
        predictors=DummyPredictors(),
        rules_engine=DummyRules(),
    )

    print("Quat 2.0 agent loop started (skeleton). Ctrl+C to stop.")
    while True:
        # Placeholder candidate
        candidate = "CCO"
        result = env.evaluate(candidate)
        print("Eval:", candidate, result)
        time.sleep(5)
