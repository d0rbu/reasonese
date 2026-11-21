// Persist provider-native items. Pi's normalized message format cannot round-trip them.
const providers = {
  anthropic: "https://api.anthropic.com/v1/messages",
  openai: "https://api.openai.com/v1/responses",
};

export function frontierProvider({ provider, model, effort }) {
  if (!Object.hasOwn(providers, provider) || !/^[a-z0-9][a-z0-9.-]+$/.test(model ?? "") ||
      !["low", "medium", "high"].includes(effort)) throw new Error("Expected a native provider, model ID and reasoning effort");
  const claude = provider === "anthropic";
  const blocks = text => [{ type: "text", text }];
  return {
    url: providers[provider],
    headers: key => claude ? { "x-api-key": key, "anthropic-version": "2023-06-01", "anthropic-beta": "thinking-binding-controls-2026-08-01" } : { authorization: `Bearer ${key}` },
    user: content => claude ? { role: "user", content: blocks(content) } : { role: "user", content },
    request(history, system, tools, limit) {
      const definitions = tools.map(({ name, description, parameters }) => claude
        ? { name, description, input_schema: parameters }
        : { type: "function", name, description, parameters, strict: false });
      if (claude) return {
        model, system, tools: definitions, messages: history, max_tokens: limit,
        thinking: { type: "adaptive", display: "omitted", block_binding: { prefix_mismatch_behavior: "error" } }, output_config: { effort },
      };
      return {
        model, instructions: system, tools: definitions, input: history, max_output_tokens: limit,
        store: false, include: ["reasoning.encrypted_content"], reasoning: { effort, context: "all_turns" },
      };
    },
    decode(response) {
      if (!response?.model || response.error) throw new Error("Invalid frontier response");
      const items = claude ? response.content : response.output;
      const status = claude ? response.stop_reason : response.status;
      if (!Array.isArray(items) || !(claude ? ["end_turn", "tool_use", "refusal"].includes(status) : status === "completed")) {
        throw new Error(`Frontier response did not complete: ${status}`);
      }
      if (response.input_transformations?.length) throw new Error("Provider changed the saved reasoning context");
      if (status === "refusal") return { saved: [{ role: "assistant", content: items }], calls: [], text: "", signed: false };
      const opaqueFields = { thinking: "signature", redacted_thinking: "data", reasoning: "encrypted_content" };
      const opaque = items.filter(item => Object.hasOwn(opaqueFields, item.type));
      if (opaque.some(item => typeof item[opaqueFields[item.type]] !== "string" || !item[opaqueFields[item.type]])) {
        throw new Error("Incomplete opaque reasoning item");
      }
      const signed = opaque.length > 0;
      const types = claude ? ["thinking", "redacted_thinking", "text", "tool_use"] : ["reasoning", "message", "function_call"];
      if (items.some(item => !types.includes(item.type))) throw new Error("Unsupported native output item");
      const calls = items.filter(item => item.type === (claude ? "tool_use" : "function_call"))
        .map(item => ({ id: claude ? item.id : item.call_id, name: item.name, args: claude ? item.input : item.arguments }));
      if ((claude && (status === "tool_use") !== Boolean(calls.length)) ||
          calls.some(call => !call.id || !call.name) || new Set(calls.map(call => call.id)).size !== calls.length) {
        throw new Error("Invalid tool calls");
      }
      const text = claude
        ? items.filter(item => item.type === "text").map(item => item.text).join("")
        : items.filter(item => item.type === "message").flatMap(item => item.content)
          .filter(item => item.type === "output_text").map(item => item.text).join("");
      const refused = !claude && items.some(item => item.type === "message" &&
        item.content.some(block => block.type === "refusal"));
      return { saved: claude ? [{ role: "assistant", content: items }] : items, calls, text: refused ? "" : text, signed };
    },
    results(results) {
      return claude ? [{ role: "user", content: results.map(({ id, text, error }) =>
        ({ type: "tool_result", tool_use_id: id, content: text, is_error: error })) }]
        : results.map(({ id, text, error }) =>
          ({ type: "function_call_output", call_id: id, output: error ? `Tool error: ${text}` : text }));
    },
  };
}
