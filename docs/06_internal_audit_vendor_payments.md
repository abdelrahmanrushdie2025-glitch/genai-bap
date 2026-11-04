# Internal Audit Report - Vendor Payment Controls

*Fictional. Internal Audit, report IA-2026-07, July 2026.*

## Summary
A review of accounts payable found weaknesses in three control areas: duplicate payments, split invoices designed to stay below the EUR 10,000 dual-approval threshold, and payments to rarely used counterparties posted outside business hours. The overall rating is "Needs improvement".

## Findings
1. Duplicate payments: several vendors were paid the same amount twice within minutes, mostly through manual SEPA transfers. Root cause: no automated duplicate check before release.
2. Split payments: series of three payments between EUR 9,500 and EUR 9,990 to the same consulting firm, Kessler & Partner Consulting, within a few hours. This pattern avoids dual approval.
3. Off-hours payments to unknown vendors: round-amount transfers (EUR 25,000 to EUR 75,000) to counterparties such as Orion Trade Ltd and Nexa Global FZE, posted at night or at weekends.
4. Extreme amounts: individual postings 15 to 60 times larger than the category norm, likely data-entry errors.

## Recommendations
Deploy automated, continuous anomaly detection that combines rule-based controls with machine learning, such as Isolation Forest. Route flagged items to a human reviewer, and measure precision and recall every quarter. Management agreed to implement this through the GenAI Business Analytics Platform by Q4 2026.
