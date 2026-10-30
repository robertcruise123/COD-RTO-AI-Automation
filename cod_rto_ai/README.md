# COD Risk Desk — RTO decision engine

A runnable prototype for the supplied Associate Software Engineer take-home assignment. It includes a machine-learning risk scorer, editable decision thresholds, plain-English reasons, a WhatsApp conversation simulator, a founder dashboard, and audit-friendly decision/AI logs.

## Run locally (under 10 minutes)

1. Install Python 3.10+.
2. Unzip the project and open a terminal in this folder.
3. Create and activate a virtual environment (optional but recommended).
4. Run:

   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

5. Open the local URL printed by Streamlit. The included `orders_sample.csv` is loaded automatically.

## What's included

- `app.py` — Streamlit dashboard, order scoring form, WhatsApp mock, model diagnostics.
- `orders_sample.csv` — the provided sample dataset.
- `config.json` — editable risk thresholds, prepaid discount and safety switches.
- `business_memo.pdf` — founder-facing business memo.
- `decision_log.md` — key choices and alternatives.
- `ai_usage_log.md` — AI use and validation notes.

## Decision policy

The model estimates the probability of RTO from order value, item count, prior order/RTO history, order time, coarse address/name/phone quality indicators, city/state, acquisition source, courier and pincode. The initial action bands are:

- `<18%`: Ship as-is, unless contact/address quality checks need confirmation.
- `18–35%`: Ship after confirmation.
- `35–55%`: Nudge to prepaid.
- `>=55%`: Ship after confirmation and manual review by default; only use hold/cancel if an operator explicitly enables auto-cancel.

These thresholds are starting hypotheses, not production-validated cutoffs. Change them in the sidebar and save to `config.json`. Address/name/phone checks are deterministic heuristics in this offline build. No LLM API is called; the README and AI log explain where an LLM could help once a provider, privacy policy and budget are approved.

## Modeling notes and limitations

- Labels are normalized (`RTO`, `DELIVERED`, case/whitespace variants). The single `IN_TRANSIT` row is excluded from supervised training.
- A time-ordered 80/20 holdout is used for a basic diagnostic, then the model is refit on all labeled rows for demo scoring. Metrics can vary and should not be treated as a reliable launch gate.
- The data has duplicate order IDs, duplicate timestamps, missing city/pincode values and a small sample. Investigate duplicate records before production training.
- The dataset contains only COD orders, so prepaid outcomes are not available to estimate the causal effect of prepaid nudges. The 4% prepaid RTO rate and 62% gross margin in the assignment brief are used only in business reasoning.
- The dashboard uses the latest date in the sample as a proxy for “yesterday”; it is not connected to Shopify or a live order feed. Savings estimates depend on explicitly editable assumptions and are not causal claims.
- The model uses historical pincode as a categorical feature; it does not automatically block a pincode. Avoid blanket blocks without adequate volume, confidence intervals and a human appeal path.

## Production next steps

Integrate Shopify order webhooks, a courier/pincode serviceability feed, an approved WhatsApp provider, verified payment webhooks, secure PII handling, model/version tracking, monitoring, opt-out handling, audit logs, and a human-review queue. Use a time-based prospective pilot and compare outcomes against a randomized or carefully matched control before claiming savings.
