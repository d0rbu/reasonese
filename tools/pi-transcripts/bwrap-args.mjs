// Preserve argument boundaries without publishing mount sources in process argv.
export function encodeBwrapArgs(args) {
  if (args.some(arg => typeof arg !== "string" || arg.includes("\0"))) throw new TypeError("Bubblewrap arguments must be strings without NUL bytes");
  return Buffer.from(args.map(arg => arg + "\0").join(""));
}
