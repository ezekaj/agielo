# Mac 2 Setup - Parallel Learning Node

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ezekaj/agielo.git
cd agielo

# 2. Install Ollama
brew install ollama
ollama serve &

# 3. Pull the model
ollama pull ministral-3:8b

# 4. Install Python dependencies
pip3 install numpy

# 5. Run the learning system
./chat.py
```

## What Mac 2 Does

Mac 2 runs as a **parallel learning node**:
- Fetches articles from GDELT, ArXiv, GitHub, ScienceDaily
- Analyzes content with Ministral
- Extracts Q&A pairs for training
- Runs benchmarks every 100 facts
- Triggers MLX fine-tuning when improved

## Keep It Running 24/7

```bash
# Prevent sleep
sudo pmset -a disablesleep 1
sudo pmset -a sleep 0
sudo pmset -a displaysleep 0

# Run with caffeinate (keeps awake while running)
caffeinate -dims ./chat.py
```

## Sync Knowledge Between Macs

### Option 1: Manual Sync (Simple)
```bash
# On Mac 2 - copy knowledge to Mac 1
scp -r ~/.cognitive_ai_knowledge/ tolga@mac1-ip:~/.cognitive_ai_knowledge_mac2/
```

### Option 2: Shared Folder (Recommended)
1. Create shared folder on iCloud/Dropbox/NAS
2. Set storage path in both Macs:
```python
# In chat.py, change:
self.trainer = SelfTrainer(storage_path="/path/to/shared/knowledge")
```

### Option 3: Git-based Sync
```bash
# Create a private repo for knowledge
# Add to crontab on both Macs:
*/30 * * * * cd ~/.cognitive_ai_knowledge && git add -A && git commit -m "sync" && git pull --rebase && git push
```

## Distributed Roles

| Mac | Role | What It Does |
|-----|------|--------------|
| Mac 1 | **Learner** | Fetches content, extracts knowledge, saves Q&A pairs |
| Mac 2 | **Trainer** | Runs MLX fine-tuning on collected data |

### Mac 2 as Dedicated Trainer
```bash
# Run only MLX training (no learning)
python3 << 'EOF'
from integrations.self_evolution import get_evolution
evo = get_evolution()
while True:
    result = evo.run_mlx_training()
    print(result)
    import time
    time.sleep(3600)  # Train every hour
EOF
```

## Monitor Progress

```bash
# Check learning stats
cat ~/.cognitive_ai_knowledge/stats.json

# Check evolution state
cat ~/.cognitive_ai_knowledge/evolution/evolution_state.json

# Count training pairs
wc -l ~/.cognitive_ai_knowledge/training_data.jsonl

# Watch live
watch -n 5 'cat ~/.cognitive_ai_knowledge/evolution/evolution_state.json'
```

## Troubleshooting

### Ollama not running
```bash
ollama serve &
```

### Model not found
```bash
ollama pull ministral-3:8b
```

### Permission denied
```bash
chmod +x chat.py
```

### Check if learning
```bash
tail -f ~/.cognitive_ai_knowledge/training_data.jsonl
```
