# AI gateway vs router

Both sit between your app and one or more LLM providers. Architecturally they look the same (HTTP proxy in front of model APIs). The difference is **ownership and trust boundary**, not protocol.

| | **Gateway** (Databricks AI Gateway, LiteLLM, Portkey, Kong AI) | **Router** (OpenRouter, etc.) |
|---|---|---|
| Who runs it | Your org (self-hosted or managed tenant) | A third-party SaaS aggregator |
| Primary user | An organization fronting its own LLM usage | An individual / small team wanting many models behind one key |
| Auth model | Internal teams get internal keys | You hand your card to one vendor; they sub-route |
| Why use it | Governance, cost attribution, policy, audit, vendor-swap | Convenience, model variety, price arbitrage |
| Routing logic | Rule-based, ops-controlled (team → model, failover order) | You name the model in the request; they forward it |
| Data residency | Stays on your tenancy | Lives on the router's servers |
| Compliance posture | "I can prove no PII left our VPC" | "I trust the router not to leak my prompts" |

**Rule of thumb:** gateway = internal control plane. Router = external aggregator.
