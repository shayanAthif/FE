# Project handoff TODO

## Server setup

- [ ] Clone or copy the repository to the server.
- [ ] Create and activate a Python virtual environment.
- [ ] Install dependencies from `requirements.txt`.
- [ ] Copy `.env.example` to `.env`.
- [ ] Set a valid `SEC_USER_AGENT` in `.env`.
- [ ] Add available market/news provider credentials to `.env`.
- [ ] Confirm `.env` is not committed.

## Preflight

- [ ] From the repository root, run `python scripts/run_pipeline.py --stage all --limit 10`.
- [ ] Confirm the smoke run exits successfully.
- [ ] Confirm the database and logs are created under the repository.
- [ ] Run the test suite with `python -m pytest -q`.

## Phase 6: scale gate

- [ ] Run `python scripts/run_phase6.py --limit 2000 --batch-size 4`.
- [ ] Review `outputs/phase6_scale_report.json`.
- [ ] Confirm the process return code is `0`.
- [ ] Confirm the resume check return code is `0`.
- [ ] Record peak RAM usage for the scale report.
- [ ] Confirm the resume check does not duplicate database rows.
- [ ] Investigate any provider errors in `logs/pipeline.log`.
- [ ] Do not start Phase 7 until all Phase 6 acceptance checks pass.

## Phase 7: full dataset

- [ ] Run `python scripts/run_phase7.py --batch-size 4`.
- [ ] If interrupted, rerun the same command without `--force`.
- [ ] Review `outputs/phase7_full_dataset_report.json`.
- [ ] Confirm the process return code is `0`.
- [ ] Record peak RAM usage for the full-dataset report.
- [ ] Confirm the resume check succeeds without duplicate rows.
- [ ] Record final transcript counts, runtime, database size, and provider failures.

## Phase 5 refresh

- [ ] Run `python scripts/run_phase5.py --skip-annotation` after Phase 7.
- [ ] Review the refreshed files in `outputs\`.
- [ ] If human labels are available, rerun with `--annotation-csv`.
- [ ] Preserve the caveat that synthetic labels are not human ground truth.

## Dashboard

- [ ] Run `streamlit run dashboard\app.py`.
- [ ] Open `http://localhost:8501`.
- [ ] Check Home, Transcript Explorer, Company Comparison, and About.
- [ ] Confirm the dashboard reads the completed SQLite database.

## Handoff evidence

- [ ] Save `outputs/phase6_scale_report.json`.
- [ ] Save `outputs/phase7_full_dataset_report.json`.
- [ ] Save the refreshed Phase 5 output files.
- [ ] Save the final test output and relevant log summary.
- [ ] Update the phase reports in `docs\reports\` with measured server results.
