import datasets

dataset = datasets.load_dataset("NovaSky-AI/SkyRL-SQL-653-data")["train"]
new_dataset = []
import subprocess
for r in dataset:
    new_dataset.append({
        "db_id": r["db_id"],
        "question": r["prompt"][-1]["content"].split("\n {question}")[-1].strip(),
        "query": r["reward_model"]["ground_truth"],
        # "data_source": r["data_source"],
        # "schema": r["prompt"][-1]["content"].split("\n {question}")[-1].strip(),
    })

# export to dataset parquet
import pandas as pd
df = pd.DataFrame(new_dataset)
df.to_parquet("/home/aiscuser/agent-lightning/examples/spider/data/skyrl_sql_653.parquet", index=False)
