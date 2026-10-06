# APRAU — group 22

Initial data understanding for the 2026/2027 APRAU assignment. Start with `docs/prd.md` for requirements and `docs/data.md` for the dataset. The introductory EDA notebook and reproducible builder are present as **unexecuted drafts**. Notebook execution was blocked by the local sandbox before results were generated.

- `docs/`: requirements, agent rules, persistent memory and data guide.
- `notebooks/`: intent → code → findings analysis notebooks.
- `reports/eda/`: reproducible profiles and cleaning evidence.
- Supplied CSV and assignment PDF remain unchanged in the project root.

Structure inspired by [the user's hockey-season-prediction project](https://github.com/AmirMasnavi/hockey-season-prediction/tree/main).

## Run the draft EDA

Install the dependencies in `requirements.txt` in a Python virtual environment, then run `python scripts/build_eda.py` from the project root. This rebuilds and executes the notebook and writes tables, figures and a derived dataset under `reports/eda/`. A local Jupyter kernel connection must be permitted. No predictive models are fitted.

The user requested a **public** repository on 6 October 2026, replacing the earlier private-repository preference. EDA completion and independent review remain pending.
