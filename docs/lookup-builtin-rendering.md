# Lookup built-in rendering artifact admission

The lookup artifact verifier supports two authenticated GGUF layouts under the
pinned Ollama `0.40.4` runtime:

| Layout | Config | Layers | Readback |
| --- | --- | --- | --- |
| Go template | `renderer` and `parser` absent or null | Exactly one template layer, no prompt layer | `/api/show.template` must match the authenticated template size and SHA-256. |
| Built-in renderer/parser | Exactly `renderer: qwen3.5` and `parser: qwen3.5` | No template or prompt layer | The authenticated config and manifest digest bind the pair; `/api/show.template` is not treated as its identity. |

Both layouts retain exact archive/runtime closure checks, manifest/config/layer
hashes, rootfs layer equality, GGUF weights, parameter count and context override
rejection, owned endpoint checks, served tag digest, `/api/show` format/family/tools,
and configured context readback. The built-in layout refuses startup profiles
that force `OLLAMA_GO_TEMPLATE`; it does not change that profile silently.

This pair is a version-bound runtime capability, not a model-name exception.
The pinned source [loads config and sets `HasGoTemplate` only for template/prompt
layers](https://github.com/ollama/ollama/blob/45e6011815c74acde6a346512856b69dd79dd22b/server/images.go#L735-L825),
[selects Ollama-rendered chat for a configured renderer or parser](https://github.com/ollama/ollama/blob/45e6011815c74acde6a346512856b69dd79dd22b/server/routes.go#L2893-L2911),
and [dispatches the configured renderer](https://github.com/ollama/ollama/blob/45e6011815c74acde6a346512856b69dd79dd22b/server/prompt.go#L135-L143).
The same revision registers the [`qwen3.5` renderer](https://github.com/ollama/ollama/blob/45e6011815c74acde6a346512856b69dd79dd22b/model/renderers/renderer.go#L71-L73)
and [parser](https://github.com/ollama/ollama/blob/45e6011815c74acde6a346512856b69dd79dd22b/model/parsers/parsers.go#L60-L61);
the parser [advertises tool support](https://github.com/ollama/ollama/blob/45e6011815c74acde6a346512856b69dd79dd22b/model/parsers/qwen35.go#L40-L45).

The pinned `/api/show` handler [constructs its response](https://github.com/ollama/ollama/blob/45e6011815c74acde6a346512856b69dd79dd22b/server/routes.go#L1921-L1934)
without populating the renderer/parser JSON fields, despite those fields existing
in the [response type](https://github.com/ollama/ollama/blob/45e6011815c74acde6a346512856b69dd79dd22b/api/types.go#L806-L824).
For rendered chat it [returns the model's default template text](https://github.com/ollama/ollama/blob/45e6011815c74acde6a346512856b69dd79dd22b/server/routes.go#L2111-L2118),
which is not a renderer provenance witness. The verified config and exact served
manifest digest supply that binding. This offline admission does not establish
model quality, runtime load success, GPU placement, or a comparative result for
`qwen3.5:9b`; those require separately authorized qualification.
