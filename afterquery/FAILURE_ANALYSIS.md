# Where frontier agents fail as tutors

Data: 5 runs x 6 salted children per teacher, Harbor + OpenCode, replayed through the simulator to recover the hidden
state at every step (`probe/replay_traces.py`; every replayed reward matches the Harbor reward exactly). Numbers are
from `FAILURE_ANALYSIS_DATA.md` (`probe/failure_analysis.py`). Interactive version: the Hidden-State Classroom page
(`viz/index.html`).

## Summary

Frontier agents (Claude Opus 5.5, GPT 6.1 Sol, three Gemini models) behave like careful, textbook teachers: they
diagnose first, review at the start of each session, never trust "I get it", and fix misconceptions when they see the
typical wrong answer. That brings them to the level of a learning-science rulebook (0.52-0.61 vs 0.56), but not to
the trained reference (0.70). The gap comes from three decisions, all about **how to spend a limited budget of turns
when the state is hidden**:

1. **Testing instead of teaching.** Agents spend 61-68% of all turns on quizzes, in every session. The reference
   starts at 46% on Monday and shifts to teaching (37% Wednesday, 27% Friday). In the last five turns before the
   exam, agents quiz 61-77% of the time; the reference 7%. A quiz gives information and some memory benefit but
   never teaches an unlearned concept, so at the end the reference's children have mastered 4.3 of 6 concepts and
   the agents' 3.3-3.8, although both reach concept 5-6.

2. **Moving on after a lucky guess.** A child who has not learned a concept still answers 1 in 4 quizzes correctly.
   Agents take a single correct answer as mastery and move to the next concept 1.4-2.0 times per child; the
   reference 0.5. Everything taught on top of that concept is then built on a missing prerequisite.

3. **The damage is done on Wednesday.** Counterfactual hand-over: replay the agent's own turns, then let the
   reference finish. Taking over after Monday recovers almost the whole gap (+0.05 to +0.11 exam points per child, the
   same as letting the reference teach all week); taking over after Wednesday recovers almost nothing (-0.01 to
   +0.02). The agents' Monday is as good as the reference's; the loss happens in the middle session, where the
   reference consolidates and pushes forward while the agents keep re-testing.

## The agent had what it needed

- The reference sees only what the agent sees (its own actions, the child's answers), through the same MCP tools.
- Giving the agent **every** simulator parameter does not help: gemini-pro-latest 0.52 -> 0.56, gemini-3.1-pro
  0.55 -> 0.52, and the mistake profile barely moves (lucky-guess move-ons 1.67 -> 1.30 and 1.97 -> 1.80).
- Six children per run means the agent can also learn from the first children; it does not change its policy.

## What does not separate agents from the reference

- Explaining on a missing prerequisite happens at a similar rate for everyone (3.1-4.1 per child), including the
  reference, which deliberately probes the next concept and then backs off.
- Misconceptions: once the signature answer appears, agents almost always run the right correction (0.00-0.03 seen
  and left uncorrected per child).
- Self-reports: agents rarely use `ask` and do not trust it.

## Caveats

- 5 runs per model (3 for GPT 6.1 Sol: two runs lost to OpenAI rate limits). Seed 3005 is a hard batch on which the
  reference loses to several agents.
- The mistake definitions use the hidden state, which the teacher cannot see; they describe outcomes of decisions,
  not mistakes the teacher could always have avoided. The hand-over test is the cleaner measure.
- The simulator is a stylised model of tutoring (see the model card in `README.md`).

## Sensitivity: is "testing instead of teaching" an artefact of our parameters?

`probe/sensitivity.py`, results in `probe/sensitivity_results.txt`. We varied the three parameters that set the
value of a quiz against an explanation: the testing-effect multiplier (1.5 / 2.5 / 4), the speed of forgetting
(half-lives x0.3 / x1 / x3) and the learn chance (x0.7 / x1 / x1.3): 27 worlds, 120 children each. In every world we
searched a family of 54 scripted teachers (correct answers needed before moving on, explanations before the first
quiz, review at session start, review turns before the exam) and compared the best one with an "agent-like" teacher
that copies the frontier agents' habits (explain once, quiz, move on after one correct answer, review at each session
start, review the last five turns; 66-69% quizzes).

- The best quiz share depends on the world: 33% to 64%, most often 50%. Fast forgetting and a weak learner push it
  up (about 60-64%); slow forgetting and a strong testing effect push it down.
- The agent-like teacher is beaten in all 27 worlds. Its regret is smallest (0.012-0.024 exam points) where
  forgetting is fast and the testing effect weak, close to our first, uncalibrated simulator, and largest
  (0.06-0.07) where forgetting is slow or children learn fast.
- So the user's concern is partly right: how much the quiz-heavy habit costs depends on the parameters, and in a
  world with very fast forgetting it is nearly optimal. But the agents keep the same habit everywhere: on the old
  fast-forgetting simulator they quizzed 64-65% of turns, on the calibrated one 61-68%. The failure is not
  "quizzing too much" in the abstract; it is not adapting the quiz/teach balance to the children in front of them,
  which the trained reference does.
- Caveat: the agent-like teacher is a scripted proxy for the agents' habits, not the agents themselves.
