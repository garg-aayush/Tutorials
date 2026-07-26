# Databricks — data concepts

Running glossary of Databricks data-side concepts (storage layout, catalog conventions, table formats, lakehouse vocabulary). Add new sections as you encounter them.

## Medallion architecture

Standard Databricks layering for catalog/schema organisation. You'll see the layer name as a suffix on catalogs (e.g. `…_bronze`, `…_silver`).

| Layer | Purpose | What you'll find |
|---|---|---|
| **bronze** | Raw ingestion, schema-on-read, minimal transformation | Raw historian / source-system dumps; rows often carry a `Properties.file_path` pointing at the source CSV/JSON |
| **silver** | Cleaned, conformed, deduplicated, type-cast, joined to dimensions | Lab data, mode flags, trip-tagged windows — the "trusted" layer most analytics reads from |
| **gold** | Business-aggregated, per-KPI | Final analytical tables; one table per KPI / dashboard / model-input |
| **research** | Experimental / ad-hoc transforms; not always production-grade | First-cut feature engineering; data-scientist sandbox. Not always present, not promoted |

Higher layers read from lower ones; never the reverse. A naive read against `bronze` will hit duplicates, non-`Good`-status rows, and unconformed schemas — that's the layer's job.
