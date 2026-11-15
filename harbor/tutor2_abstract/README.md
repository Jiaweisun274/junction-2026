# Harbor task: AfterQuery tutor2, abstract skin ("Node Lattice")

Same simulator, same random draws and same scoring as `../tutor2`; only the story and the tool names change
(child -> board, concept -> node, explain -> charge, quiz -> probe, correct -> repair, ask -> ping,
misconception -> fault, session -> window, exam -> evaluation; answers become symbols). The mapping is in
`environment/mcp-server/skin.py`. A run on seed S is therefore paired one-to-one with the tutoring run on seed S,
which isolates the effect of the tutoring story (and the model's prior knowledge about teaching) on decisions.

Run: `TASK=tutor2_abstract harbor/run_seeds.sh opencode <model> <tag> <seed...>` from the repository root.
