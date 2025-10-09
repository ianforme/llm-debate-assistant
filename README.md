# llm-debate-assistant
基于大语言模型的辩论比赛备赛助手. Powered by GPT 5

---

## Table of Contents
- [llm-debate-assistant](#llm-debate-assistant)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
  - [Run the App](#run-the-app)

## Getting Started
This project uses `poetry` to manage dependencies

```bash
poetry config virtualenvs.in-project true
```


Then install the dependencies
```bash
poetry install
```

Then activate the virtual environment
```bash
source .venv/bin/activate
```

Note: to run the llms, set `is_demo=False` in `config.py`. Otherwise the demo UI will load the generated outputs from the previous run

---

## Run the App

To generate the debate content, you can either run the script:
```bash
python -m examples.run_debate
```
Or run the cells in the Jupyter notebook: [`notebook/scratchpad.ipynb`](notebook/scratchpad.ipynb).

To view the debate in the UI:
```bash
streamlit run frontend/app.py
```
