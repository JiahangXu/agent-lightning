#!/bin/bash

# ====== Config ======
BERT_LOG="logs/bert.log"
MEM_LOG="logs/mem.log"
BERT_PORT=8000
MEM_PORT=8001

# ====== Kill any process using the ports ======
sudo lsof -t -i :$BERT_PORT | xargs -r sudo kill -9
sudo lsof -t -i :$MEM_PORT | xargs -r sudo kill -9

# ====== Create log directory ======
mkdir -p logs

# ====== Start child servers in background ======
nohup python algorithms/empo2/server_bert.py > "$BERT_LOG" 2>&1 &
nohup python algorithms/empo2/server_mem.py > "$MEM_LOG" 2>&1 &

echo "✅ Child servers started"
echo "📜 BERT log: $BERT_LOG"
echo "📜 MEM log:  $MEM_LOG"