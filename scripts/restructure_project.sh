#!/bin/bash
# Script to restructure DeepSculpt project
# This will:
# 1. Move deepsculpt_legacy to boilerplate/
# 2. Rename deepsculpt_v2 to deepsculpt
# 3. Update all imports

set -e  # Exit on error

echo "🔄 Restructuring DeepSculpt project..."
echo ""

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Step 1: Move legacy to boilerplate
echo "📦 Step 1: Moving deepsculpt_legacy to boilerplate/"
if [ -d "deepsculpt_legacy" ]; then
    mv deepsculpt_legacy boilerplate/deepsculpt_legacy
    echo "✅ Moved deepsculpt_legacy to boilerplate/"
else
    echo "⚠️  deepsculpt_legacy not found, skipping"
fi

# Step 2: Rename deepsculpt_v2 to deepsculpt_new (temporary)
echo ""
echo "📝 Step 2: Renaming deepsculpt_v2 to deepsculpt"
if [ -d "deepsculpt_v2" ]; then
    # First rename to temp name to avoid conflicts
    mv deepsculpt_v2 deepsculpt_new
    echo "✅ Renamed to temporary name"
    
    # Then rename to final name
    mv deepsculpt_new deepsculpt
    echo "✅ Renamed to deepsculpt"
else
    echo "❌ deepsculpt_v2 not found!"
    exit 1
fi

# Step 3: Update imports in Python files
echo ""
echo "🔧 Step 3: Updating imports in Python files..."

# Find all Python files and update imports
find deepsculpt -type f -name "*.py" -exec sed -i '' 's/from deepsculpt_v2\./from deepsculpt./g' {} +
find deepsculpt -type f -name "*.py" -exec sed -i '' 's/import deepsculpt_v2/import deepsculpt/g' {} +

echo "✅ Updated Python imports"

# Step 4: Update imports in notebooks
echo ""
echo "📓 Step 4: Updating imports in notebooks..."

find deepsculpt/notebooks -type f -name "*.ipynb" -exec sed -i '' 's/deepsculpt_v2/deepsculpt/g' {} +

echo "✅ Updated notebook imports"

# Step 5: Update path references in scripts
echo ""
echo "📜 Step 5: Updating path references..."

find deepsculpt/scripts -type f -name "*.py" -exec sed -i '' 's/deepsculpt_v2/deepsculpt/g' {} +

echo "✅ Updated script paths"

echo ""
echo "🎉 Restructuring complete!"
echo ""
echo "Summary:"
echo "  ✅ Moved deepsculpt_legacy → boilerplate/deepsculpt_legacy"
echo "  ✅ Renamed deepsculpt_v2 → deepsculpt"
echo "  ✅ Updated all imports and paths"
echo ""
echo "Next steps:"
echo "  1. Test the imports: python -c 'import deepsculpt'"
echo "  2. Run tests if available"
echo "  3. Commit changes: git add -A && git commit -m 'Restructure: move legacy to boilerplate, rename v2 to main'"
