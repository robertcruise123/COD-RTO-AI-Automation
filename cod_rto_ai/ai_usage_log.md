# AI usage log

## AI/tools used

- ChatGPT was used to help structure the application, derive a baseline decision policy, write code/documentation, and draft the business memo.
- Python, pandas and scikit-learn are used for deterministic data inspection, feature engineering, supervised training and metrics.
- Streamlit is used for the local interactive dashboard and mock WhatsApp simulator.
- No external LLM API is called by the app. No customer data is sent to a third party by this prototype.

## Examples of weak output/risk and how it was checked

1. **Risk of treating every status as a valid class.** Case and whitespace variants (`RTO`, `Rto`, `RTO `, `delivered`) could silently fragment labels. The preprocessing normalizes case/whitespace and explicitly excludes `IN_TRANSIT` from training.
2. **Risk of overstating historical savings.** A retrospective rule can appear to “save” an order without evidence the customer would have refused delivery or that the intervention changed the outcome. The dashboard labels savings as scenario estimates, exposes assumptions, and avoids causal language.
3. **Risk of auto-cancelling first-time customers.** A high model score alone could harm growth, especially given the brief's acquisition cost concern. Automatic cancellation is off by default; high-risk orders go to confirmation/manual review unless an operator opts in to a controlled policy.

## LLM boundary

An LLM can be useful for interpreting a Hinglish address, classifying whether a landmark description is understandable, or turning a free-text customer reply into a suggested intent. It should not be used for arithmetic, risk thresholds, payment verification, or as the sole basis for cancelling an order. Any future LLM call should have a strict schema, confidence/abstention behavior, privacy review, logging, and a deterministic fallback.
