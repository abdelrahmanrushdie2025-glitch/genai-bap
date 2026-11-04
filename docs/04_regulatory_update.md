# Regulatory Update - AI, Operational Resilience and Data Protection

*Fictional internal compliance briefing, Compliance & Risk, August 2026. Not legal advice.*

## EU AI Act
The EU Artificial Intelligence Act entered into force in August 2024, and its obligations apply in stages. AI systems used for creditworthiness assessment of individuals are classified as high-risk. AMG's analytics platform supports internal reporting and is not used for credit decisions. It is therefore expected to fall outside the high-risk category, but transparency, human oversight and logging are still applied as good practice.

## DORA (Digital Operational Resilience Act)
DORA has applied to EU financial entities since 17 January 2025. It requires ICT risk management, incident reporting, resilience testing and oversight of third-party ICT providers, including cloud providers. Our cloud contracts must include exit strategies, audit rights and clear data-location commitments.

## GDPR
Personal data in ledgers (for example employee expense claims) must be minimised. The analytics platform only processes pseudonymised counterparty identifiers. Prompts sent to external LLM APIs must not contain personal data. Production deployments should use an EU-hosted model endpoint with a data-processing agreement.

## Required controls for the GenAI platform
- Read-only database access for AI-generated SQL; block any data-modifying statements.
- Show sources and SQL with every answer (explainability).
- Log prompts, model versions and token usage for audit.
- Hide evaluation labels and sensitive columns from the model.
- Run a quarterly review of model accuracy and anomaly-detection precision.
