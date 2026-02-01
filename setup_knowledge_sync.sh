#!/bin/bash
# Setup Git-based knowledge sync between Macs
# Run this on BOTH Macs

set -e

KNOWLEDGE_DIR="$HOME/.cognitive_ai_knowledge"
REPO_URL="${1:-}"  # Pass repo URL as argument, or it will create one

echo "=== Knowledge Sync Setup ==="

# Create knowledge directory if not exists
mkdir -p "$KNOWLEDGE_DIR"
cd "$KNOWLEDGE_DIR"

# Initialize git if needed
if [ ! -d ".git" ]; then
    echo "Initializing git repo..."
    git init

    # Create .gitignore
    cat > .gitignore << 'EOF'
*.lock
*.tmp
__pycache__/
EOF

    git add -A
    git commit -m "Initial knowledge base" || true
fi

# Set up remote if provided
if [ -n "$REPO_URL" ]; then
    echo "Setting up remote: $REPO_URL"
    git remote remove origin 2>/dev/null || true
    git remote add origin "$REPO_URL"
    git branch -M main
    git push -u origin main --force
    echo "Remote configured!"
fi

# Create sync script
SYNC_SCRIPT="$KNOWLEDGE_DIR/sync.sh"
cat > "$SYNC_SCRIPT" << 'EOF'
#!/bin/bash
cd ~/.cognitive_ai_knowledge
git add -A
git diff --cached --quiet || git commit -m "sync $(date +%Y%m%d-%H%M%S)"
git pull --rebase origin main 2>/dev/null || true
git push origin main 2>/dev/null || true
EOF
chmod +x "$SYNC_SCRIPT"

# Add to crontab (every 10 minutes)
CRON_CMD="*/10 * * * * $SYNC_SCRIPT >> $KNOWLEDGE_DIR/sync.log 2>&1"
(crontab -l 2>/dev/null | grep -v "cognitive_ai_knowledge/sync.sh"; echo "$CRON_CMD") | crontab -

echo ""
echo "=== Setup Complete ==="
echo "Knowledge dir: $KNOWLEDGE_DIR"
echo "Sync script: $SYNC_SCRIPT"
echo "Cron: Every 10 minutes"
echo ""
echo "To manually sync: $SYNC_SCRIPT"
echo ""
echo "Next steps:"
echo "1. Create a private GitHub repo for knowledge"
echo "2. Run: $0 git@github.com:YOUR_USER/knowledge.git"
echo "3. Run the same on Mac 2"
