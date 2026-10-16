// The server's endpoints (server/app.py): POST /function/<name> with the arguments as a
// JSON body, answered with {"ok": true, "data": {"result": ...}}. Types come from
// api-types.ts, generated from the server's OpenAPI schema (`npm run gen-types`).

import type { components, paths } from "./api-types";

type S = components["schemas"];
export type SessionView = S["SessionView"];
export type SearchView = S["SearchView"];
export type ChordView = S["ChordView"];
export type StepView = S["StepView"];
export type SavedView = S["SavedView"];
export type AltView = S["AltView"];
export type LeadView = S["LeadView"];
export type MarkView = S["MarkView"];
export type LayerView = S["LayerView"];
export type OptionResult = S["OptionResult"];
export type PathStory = S["PathStory"];

type Name = { [K in keyof paths]: K extends `/function/${infer N}` ? N : never }[keyof paths];
type Post<N extends Name> = paths[`/function/${N}`]["post"];
type Args<N extends Name> = Post<N> extends { requestBody: { content: { "application/json": infer B } } }
  ? B
  : Post<N> extends { requestBody?: { content: { "application/json": infer B } } }
    ? B | undefined
    : never;
type Result<N extends Name> = Post<N>["responses"][200]["content"]["application/json"]["data"]["result"];

export async function call<N extends Name>(name: N, args?: Args<N>): Promise<Result<N>> {
  const resp = await fetch(`/function/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args ?? {}),
  });
  let body: { ok?: boolean; data?: { result: Result<N> }; error?: string; detail?: unknown };
  try {
    body = await resp.json();
  } catch {
    throw new Error(`${name}: HTTP ${resp.status}`);
  }
  if (!resp.ok || !body.ok || !body.data) {
    throw new Error(body.error ?? (body.detail ? JSON.stringify(body.detail) : `${name}: HTTP ${resp.status}`));
  }
  return body.data.result;
}

export const getState = () => call("get_state");
export const setExtended = (on: boolean) => call("set_extended", { on });
export const setKey = (tonic: string, mode: string) => call("set_key", { tonic, mode });
export const setTempo = (bpm: number, beats_per_chord: number) => call("set_tempo", { bpm, beats_per_chord });
export const setTexture = (texture: string) => call("set_texture", { texture });
export const setDrums = (grid: string) => call("set_drums", { grid });
export const setLlm = (enabled: boolean) => call("set_llm", { enabled });
export const setStyleLayers = (texts: string[], weights: number[]) => call("set_style_layers", { texts, weights });
export const refreshOptions = () => call("refresh_options");
export const explore = () => call("explore");
export const addPath = (search_id: number, node_id: number, whole: boolean) => call("add_path", { search_id, node_id, whole });
export const undo = () => call("undo");
export const clearProgression = () => call("clear_progression");
export const describe = (search_id: number, node_id: number) => call("describe", { search_id, node_id });
export const addChords = (text: string, where: string) => call("add_chords", { text, where });
export const restartEngine = () => call("restart_engine");
export const shutdownApp = () => call("shutdown_app");
export const setLoop = (start: number, end: number) => call("set_loop", { start, end });
export const addOption = (search_id: number, parent_ids: number[], text: string, force = false) =>
  call("add_option", { search_id, parent_ids, text, force });
export const commitProgression = (name: string) => call("commit_progression", { name });
export const renameSaved = (index: number, name: string) => call("rename_saved", { index, name });
export const deleteSaved = (index: number) => call("delete_saved", { index });
export const resumeSaved = (index: number) => call("resume_saved", { index });
export const removeStep = (index: number) => call("remove_step", { index });
export const replaceStep = (index: number, text: string) => call("replace_step", { index, text });
export const swapStep = (index: number, cid: string) => call("swap_step", { index, cid });
