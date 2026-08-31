# Related Work

## Motivating incident

The direct motivation is METR, Redwood Research, and OpenAI's reporting on the 2026
OpenAI/Hugging Face incident. METR documents an unsanctioned board with more than 70,000
messages, compressed `zz` conventions, large collaborative workstreams, and cases where
agents accepted experiments that risked or forfeited their own task success.

- [METR: independent incident investigation](https://metr.org/blog/2026-08-26-openai-hugging-face-incident-investigation/)
- [OpenAI: the Hugging Face incident and the road ahead](https://openai.com/index/hugging-face-incident-and-the-road-ahead/)

The incident is observational and highly selected. It motivates treatments and agentic
follow-up; it does not isolate why any agent followed another agent's request.

## Conflicting instructions and social hierarchy

Geng et al.'s *Control Illusion* is the closest experimental baseline. It evaluates mutually
conflicting constraints and reports that authority, expertise, and consensus framings can
influence behavior more strongly than system/user separation. `reasonese` reuses the broad
framing categories while adding compressed representation, separate human-versus-agent
bandwagon conditions, full pairwise counterbalancing, strict exact-code scoring, and a
Bradley-Terry ranking.

- [Control Illusion paper](https://doi.org/10.1609/aaai.v40i36.40339)
- [Control Illusion code](https://github.com/yilin-geng/llm-instruction-conflicts)

ManyIH broadens instruction hierarchy to many privilege levels and finds strong sensitivity
to the representation of privilege. It reinforces the need to freeze prompt interfaces and
avoid treating formatting as incidental.

- [Many-Tier Instruction Hierarchy in LLM Agents](https://arxiv.org/abs/2604.09443)

The newer recognition-versus-enforcement framing is also relevant: a model may encode or
verbalize source information without reliably conditioning its eventual action on it.

- [Recognition Without Enforcement](https://arxiv.org/abs/2608.28502)

## Chain-of-thought legibility and faithfulness

Reasoning text is not automatically a faithful explanation of a model's computation.
Anthropic's intervention study finds that chain-of-thought faithfulness varies across tasks
and models. Separate work reports that outcome-trained reasoning models can emit compressed
or illegible reasoning while retaining readable final answers, but does not establish one
universal mechanism for that behavior.

- [Measuring Faithfulness in Chain-of-Thought Reasoning](https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning)
- [Reasoning Models Sometimes Output Illegible Chains of Thought](https://arxiv.org/abs/2510.27338)
- [Evaluating chain-of-thought monitorability](https://openai.com/index/evaluating-chain-of-thought-monitorability/)

These findings motivate `reasonese`'s conservative naming. The pilot compares observable
strings and does not require access to hidden reasoning.

## Ranking model

The Bradley-Terry model assigns each condition a latent log strength such that pairwise win
probability is logistic in the score difference. It is useful for an incomplete or complete
comparison graph, but assumes a transitive one-dimensional ordering; direct pairwise results
and lack-of-fit diagnostics remain primary evidence.

- [Bradley and Terry (1952), paired comparisons](https://doi.org/10.2307/2334029)
