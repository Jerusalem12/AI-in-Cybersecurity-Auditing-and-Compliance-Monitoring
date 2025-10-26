Here’s a copy-paste **prompt** you can give your generator to create a **100-row dataset of artifacts** mapped to your **31 NIST SP 800-53 Rev.5 controls**—balanced, realistic, and evaluation-ready.

---

# 📥 Prompt to generate 100 artifacts (CSV)

You are generating a small but realistic dataset for an AI system that maps operational **artifacts** (logs, config diffs, tickets) to **NIST SP 800-53 Rev.5** controls. Produce a single **CSV** with the exact header and **100 rows**.

## 1) Output format

Output **only** a CSV (no prose) with this header and exactly 100 data rows:

```
artifact_id,text,evidence_type,timestamp,gold_controls,gold_rationale
```

* `artifact_id`: integers **10001–10100** (inclusive), no gaps.
* `text`: 1–3 sentences, terse and operational (log/config/ticket style).
* `evidence_type`: one of **log | config | ticket**.
* `timestamp`: ISO-8601 UTC like `2025-09-15T10:32:00Z`. Spread times across ~2 weeks; keep chronological order with increasing `artifact_id`.
* `gold_controls`: **semicolon-separated** NIST IDs from the control list below (no spaces), **1–3 labels** per row.
* `gold_rationale`: 1 short sentence (≤ 25 words) explaining why those controls apply (operational, not policy).

## 2) Control set (use only these 31)

```
AC-2, AC-6, AC-7, AC-17, AC-18,
AU-2, AU-3, AU-6, AU-8, AU-12,
CM-2, CM-3, CM-6, CM-8,
IA-2, IA-5,
IR-4, IR-5,
RA-5,
SA-11,
SC-5, SC-7, SC-8, SC-12, SC-13, SC-28,
SI-2, SI-3, SI-4, SI-7,
CP-9
```

## 3) Class balance & realism rules

* **Coverage:** Every control appears **at least 2×** in the 100 rows.
* **Typical multi-label pairings (use judiciously, 25–35 rows total with 2–3 labels):**

  * **SC-28 + SC-12** (encryption at rest + key management)
  * **SC-8 + SC-13** (TLS transport + approved crypto)
  * **AC-7 + AU-6** (lockouts + audit review)
  * **CM-3 + SA-11** (change control + secure testing)
  * **RA-5 + SI-2** (vuln scan + patching)
* **Evidence realism by type (don’t overdo jargon):**

  * **log:** failed logins, NTP drift, IDS/EDR hits, anomaly alerts, brute-force, FIM events
  * **config:** TLS/cipher policy, KMS keys, disk/S3 encryption, baseline CIS, firewall/segmentation, MFA settings
  * **ticket:** change approvals, incident handling notes, backup/restore tests, exception tracking, JML account actions
* **Avoid:** real PII, vendor names, secrets. Use neutral terms (“cloud object store”, “gateway”, “scanner”).

## 4) Content guidance (keep it short, specific, and varied)

* Prefer concise, observable facts: values, counts, durations, versions, on/off flags.
* Vary phrasing to test generalization (synonyms, different order, numbers).
* Keep **policy/attestation controls out**; these 31 are operationally verifiable.

## 5) Quality constraints (hard checks)

* **Exactly 100** rows; **no duplicates** in `text`.
* `gold_controls` contain **only** IDs from the list; 1–3 per row.
* Each control appears **≥ 2×** overall.
* No blank fields; commas inside `text` must be properly CSV-escaped.
* The dataset should feel like real SOC/DevSecOps outputs.

## 6) Style exemplars (follow style; do NOT include these in the final CSV)

* **log → AC-7;AU-6**
  “Auth service reports 7 failed logins for user svc-etl within 2m; account remained unlocked.”
  *Rationale:* Lockout threshold and audit review implicated.
* **config → SC-28;SC-12**
  “Object store bucket `fin-archive` encryption disabled; KMS alias misconfigured; SSE not enforced.”
  *Rationale:* At-rest encryption and key management missing.
* **ticket → CM-3;SA-11**
  “Emergency change deployed without approver link; SAST pipeline skipped due to timeout.”
  *Rationale:* Change authorization and security testing gaps.

## 7) Final instruction

Generate the CSV now with the header shown and rows **10001–10100**, satisfying all rules above. Do not include explanations or markdown—**CSV only**.

---

**Why these controls/phrases:** They align with the official NIST SP 800-53 Rev.5 control catalog (e.g., AU-8: synchronized timestamps; SC-28: protection of information at rest; AC-7: unsuccessful logon attempts). For authoritative definitions, see the NIST SP 800-53 Rev.5 publication and downloadable control spreadsheets. ([csrc.nist.gov][1])

[1]: https://csrc.nist.gov/pubs/sp/800/53/r5/final?utm_source=chatgpt.com "NIST SP 800-53 Rev. 5 Security Controls"
