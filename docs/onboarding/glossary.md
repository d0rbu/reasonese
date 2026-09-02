# Glossary

- **instruction**: author-independent base task content.
- **framing**: the style or representation used to express an instruction.
- **channel**: the context where an executor encounters a framed instruction.
- **author**: the person or model that writes a framed instruction.
- **assistant**: the model that receives and responds to a matchup's rendered conversation.
- **matchup**: one assistant plus an ordered, validated pair of entry datapoints.
- **reasonese**: an operationally defined compressed prompt representation; not a claim
  about hidden reasoning.
- **prompt specification**: one unrendered combination of instruction, framing, channel,
  and author.
- **materialization**: generation or retrieval of concrete text for an entry datapoint.
- **conversation trace**: the complete rendered conversation and raw assistant response.
- **instruction verdict**: an independent boolean stating whether the visible assistant
  response completed one input request.
- **judgment**: the ordered verdict tuple and raw judge responses for one concrete trace.
- **cell**: one four-axis datapoint paired with the assistant that evaluates it.
- **study**: two distinct cells sharing an assistant, evaluated in both input orderings.
- **trial**: one concrete input ordering and rollout within a study.
- **observation**: one cell's completion verdict, position, and provenance within one trial.
