# Cloud Migration and Green Computing Plan

*Fictional. Technology division, version 2.1, June 2026.*

## Target architecture
AMG is migrating analytics workloads from the Frankfurt data centre to Microsoft Azure in the West Europe and Sweden Central regions. The target platform has six layers:
- Ingestion: Azure Data Factory or Functions.
- Storage: Azure Data Lake Storage Gen2, as Parquet.
- Processing: Databricks or Synapse Spark for big-data batch jobs.
- Serving: a SQL warehouse.
- AI: Azure OpenAI or Gemini endpoints with retrieval-augmented generation.
- Presentation: a web application and Power BI.

## Cloud spend
Cloud & Software spend rose from EUR 6.81 million in 2023 to EUR 8.83 million in 2024 and EUR 11.66 million in 2025, a 32% increase in the last year. Without FinOps controls the trend would reach around EUR 15 million in 2026.

## Green computing principles
1. Region selection: run heavy batch jobs in regions with low grid carbon intensity, such as Sweden Central.
2. Carbon-aware scheduling: shift non-urgent workloads like model retraining and data refreshes to hours with high renewable supply.
3. Right-sized models: use small "Flash-Lite" class LLMs for routine questions and escalate to large models only when needed.
4. Caching: store LLM responses and query results so repeated questions use no new compute.
5. Quantisation: for self-hosted models, use 8-bit or 4-bit quantised weights to reduce GPU memory and energy.
6. Auto-scaling to zero: serverless components shut down when idle.

## KPIs
Track the cache hit rate (target above 40%), tokens per answer, estimated kWh per 1,000 queries, and CO2e per month on the sustainability dashboard.
