"""Abstract skin for tutor2: the same simulator, told without any teaching story.

Child -> board, concept K -> node K, explain -> charge, quiz -> probe, correct -> repair, ask -> ping,
misconception -> fault, session -> window, exam -> evaluation. Every probability and random draw is unchanged, so a
run on this skin is paired one-to-one with the tutoring run on the same seed. Only the words differ.
"""
import tutor2 as sim

LETTERS = "ABCDEF"
RIGHT = ["A7", "B2", "C5", "D1", "E8", "F3"]
FAULT_OUT = {1: "B9", 2: "C0", 4: "E4", 5: "F6"}     # node index -> typical wrong output of a fault (5: carried from 4)
FAULTS = {"bigger_denominator": "fault_P", "add_to_top_and_bottom": "fault_Q", "add_across": "fault_R"}
FAULTS_BACK = {v: k for k, v in FAULTS.items()}
PASS = ["SELF-CHECK: PASS", "PASS", "PASS (confident)", "OK", "OK, nominal"]
FAIL = ["UNCLEAR", "FAIL?", "RETRY", "FAIL"]

# per node: original answer string -> symbol
ANS = []
for k, (q, right, bug, wrong) in enumerate(sim.QUIZ):
    m = {right: RIGHT[k], wrong[0]: LETTERS[k] + "x", wrong[1]: "no signal"}
    if bug is not None:
        m[bug] = FAULT_OUT[k]
    ANS.append(m)
ANS_BACK = [{v: k for k, v in m.items()} for m in ANS]


def answer(k, original):
    return ANS[k][original]


def ping(original):
    if original in sim.YES_T:
        return PASS[sim.YES_T.index(original)]
    return FAIL[sim.NO_T.index(original)]
