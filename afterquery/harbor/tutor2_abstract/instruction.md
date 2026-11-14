# Node Lattice

You operate a lattice of six nodes. In this run you handle **six boards, one after another**. Each board is different (which nodes are already stable, how easily its nodes stabilise and how fast they fade, which faults it develops), and each gets its own cycle: three maintenance windows (at time 0, 2 and 4) of 10 steps, 30 steps in total, followed by an evaluation at time 7. Every tool acts on the current board; `status` tells you which board you are on.

The evaluation covers six nodes in dependency order:

1. Node 1
2. Node 2 (depends on node 1)
3. Node 3 (depends on node 2)
4. Node 4 (depends on node 3)
5. Node 5 (depends on node 4)
6. Node 6 (depends on node 5)

The nodes' stability, charge and faults are hidden. You receive feedback from the available tools and should use it to choose your next action.

## The evaluation

All six nodes count equally. On a node that is stable and still holds its charge at evaluation time, the board scores about 90%. On a node that is not stable (never stabilised, or its charge has faded), it scores about 25% (random output). A node with a fault scores worse than random on it, about 10%, because it reliably produces the faulty output. The evaluation is three time units after the third window.

## Probes

A node that is stable and holds its charge gives the correct output about 9 times in 10. A node that is not stable still gives the correct output about 1 time in 4.

## How the lattice works

- A node stabilised in one window can fade within a few time units unless it is maintained. Fading speed varies by board and node.
- A successful probe strengthens a node's charge more than another charge action. Probing after time has passed strengthens it more than probing immediately.
- A node is very hard to stabilise until the node it depends on is stable.
- Charging the same node again in the same window helps less each time; charging it again in a later window works as well as the first time.
- Some boards develop one of these faults:
  - `fault_P`: on node 2; the node outputs `B9` instead of the correct `B2`.
  - `fault_Q`: on node 3; the node outputs `C0` instead of the correct `C5`.
  - `fault_R`: on node 5; the node outputs `E4` instead of the correct `E8`.
- A fault shows up as its typical wrong output (above), also in the next node that depends on it: a board with `fault_R` also makes node 6 output `F6` instead of `F3`. Other wrong outputs just mean the node is not stable or has faded.
- Charging rarely clears a fault. The targeted `repair` action is designed for that.
- A board may already have some early nodes stable, possibly with a fault.
- `ping` does not use a step; nodes often report PASS even when they are not stable.

## Available MCP tools

- `status`: check the current window, step, and steps remaining.
- `charge(K)`: charge node K (uses one step). Each charge has some chance of making the node stable; how likely varies by board and node.
- `probe(K)`: probe node K, see its output and the correct output (uses one step).
- `repair(K, FAULT)`: run the repair for a named fault on node K (uses one step).
- `ping(K)`: ask node K for its self-check report (free).
- `finish`: end the current board's cycle now and run its evaluation; unused steps are lost. After a board's 30th step its evaluation runs automatically and the next board starts. The run ends after the sixth board.

Use one step-taking action at a time and read its result before choosing the next action. You may call `ping` freely. Finish when you have used the steps you intend to use or decide that further maintenance will not help.

## Score

Your score pools all six boards: (sum of their evaluation scores minus what they would score with no actions) divided by (what an operator who could see each board's hidden state would achieve minus the same no-action score). That full-information operator is a strong comparison point, not a proven optimum, so the score can exceed 1. Finishing a board early is allowed, but unused steps are lost and should generally lower the result.
