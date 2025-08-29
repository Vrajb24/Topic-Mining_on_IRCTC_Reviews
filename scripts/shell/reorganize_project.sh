#!/bin/bash

# Reorganize Project Structure for Cleaner Git Repository
echo "🔧 Reorganizing Project Structure"
echo "=================================="

# Create scripts directory
echo "📁 Creating scripts directory..."
mkdir -p scripts

# Move run scripts to scripts directory
echo "📦 Moving pipeline scripts..."
git mv run_batch_scraping.py scripts/ 2>/dev/null || mv run_batch_scraping.py scripts/
git mv run_full_analysis.py scripts/ 2>/dev/null || mv run_full_analysis.py scripts/
git mv run_full_scraping.py scripts/ 2>/dev/null || mv run_full_scraping.py scripts/
git mv run_lda_analysis.py scripts/ 2>/dev/null || mv run_lda_analysis.py scripts/

# Move utility scripts
echo "📦 Moving utility scripts..."
git mv generate_report.py scripts/ 2>/dev/null || mv generate_report.py scripts/
git mv preprocess_all.py scripts/ 2>/dev/null || mv preprocess_all.py scripts/

# Keep these in root for easy access
echo "📌 Keeping essential scripts in root..."
# run_complete_pipeline.py - Main automation (keep in root)
# streamlit_app.py - Main app (keep in root)

# Remove duplicate/unnecessary files
echo "🗑️ Removing unnecessary files..."
rm -f dashboard_feedback_loop.py 2>/dev/null
rm -f capture_dashboard.py 2>/dev/null
rm -f test_*.py 2>/dev/null
rm -f prepare_deployment.sh 2>/dev/null
rm -f prepare_for_github.sh 2>/dev/null

# Clean up shell scripts
echo "📦 Organizing shell scripts..."
mkdir -p scripts/shell
mv *.sh scripts/shell/ 2>/dev/null || true

# Remove deploy directory (already in streamlit-deploy)
echo "🗑️ Cleaning duplicate directories..."
git rm -r deploy/ 2>/dev/null || rm -rf deploy/

# Update imports in run_complete_pipeline.py if needed
echo "🔧 Updating import paths..."
# No updates needed as scripts are standalone

# Create README in scripts directory
echo "📝 Creating scripts README..."
cat > scripts/README.md << 'EOF'
# Scripts Directory

This directory contains various utility and pipeline scripts for the IRCTC Review Analysis project.

## Main Scripts (in root):
- `run_complete_pipeline.py` - Complete automation pipeline
- `streamlit_app.py` - Main dashboard application

## Pipeline Scripts:
- `run_batch_scraping.py` - Batch scraping execution
- `run_full_analysis.py` - Complete analysis pipeline
- `run_full_scraping.py` - Full scraping pipeline
- `run_lda_analysis.py` - LDA topic modeling

## Utility Scripts:
- `generate_report.py` - Report generation
- `preprocess_all.py` - Text preprocessing

## Usage:
All scripts can be run from the project root:
```bash
python scripts/run_full_analysis.py
```
EOF

echo "✅ Reorganization complete!"
echo ""
echo "📊 New Structure:"
echo "  Root/"
echo "  ├── run_complete_pipeline.py (main)"
echo "  ├── streamlit_app.py (dashboard)"
echo "  ├── scripts/"
echo "  │   ├── run_batch_scraping.py"
echo "  │   ├── run_full_analysis.py"
echo "  │   ├── run_full_scraping.py"
echo "  │   ├── run_lda_analysis.py"
echo "  │   ├── generate_report.py"
echo "  │   └── preprocess_all.py"
echo "  └── src/ (unchanged)"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Commit: git add . && git commit -m 'Reorganize scripts into dedicated directory'"
echo "3. Push: git push origin main"