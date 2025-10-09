# llm-debate-assistant
基于大语言模型的辩论比赛备赛助手. Powered by GPT 5

# Installation
1. Python 3.11 and above
2. `pip install -r requirements.txt`
3. update the openai project, org and api keys in config
4. to run in streamlit UI: `streamlit run app.py`

Note: to run the llms, set `is_demo=False` in `config.py`. Otherwise the demo UI will load the generated outputs from the previous run
