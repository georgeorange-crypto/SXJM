"""Synthetic smoke only: validate SMDP option value ordering."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT), str(Path(__file__).resolve().parents[1]/"src")]
from way4.core.events import OptionTransition, transition_dataset
from way4.rl.smdp import TabularSMDPLearner

def main():
    rows=[]
    for i in range(50):
        rows.append(OptionTransition(((1,"UNKNOWN"),), "SCAN", (0.,0.), (1,), 5.+i%3, 1,
                                     ((1,"DETECTED"),), False, False))
        rows.append(OptionTransition(((1,"DETECTED"),), "CLEAR", (0.,0.), (1,), 5.+i%2, 1,
                                     ((1,"CLEARED"),), True, True))
    learner=TabularSMDPLearner(); values=learner.fit(transition_dataset(rows))
    payload={"kind":"smdp-learner-smoke-only","n_transitions":len(rows),
             "values":values,"clear_beats_scan":values["CLEAR"]>values["SCAN"],
             "real_rollout_evidence":False}
    out=Path(__file__).resolve().parents[1]/"results"/"smdp_learner_smoke.json"
    out.write_text(json.dumps(payload,indent=2),encoding="utf-8"); print(json.dumps(payload,indent=2))
if __name__ == "__main__": main()
