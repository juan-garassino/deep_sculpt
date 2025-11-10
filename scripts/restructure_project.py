#!/usr/bin/env python3
"""
Script to restructure DeepSculpt project.
This will:
1. Move deepsculpt_legacy to boilerplate/
2. Rename deepsculpt_v2 to deepsculpt
3. Update all imports
"""

import os
import shutil
import re
from pathlib import Path


def update_file_content(file_path: Path, old_text: str, new_text: str):
    """Update text in a file."""
    try:
        content = file_path.read_text()
        if old_text in content:
            new_content = content.replace(old_text, new_text)
            file_path.write_text(new_content)
            return True
        return False
    except Exception as e:
        print(f"  ⚠️  Error updating {file_path}: {e}")
        return False


def main():
    print("🔄 Restructuring DeepSculpt project...")
    print()
    
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    os.chdir(project_root)
    
    # Step 1: Move legacy to boilerplate
    print("📦 Step 1: Moving deepsculpt_legacy to boilerplate/")
    legacy_dir = project_root / "deepsculpt_legacy"
    boilerplate_dir = project_root / "boilerplate"
    
    if legacy_dir.exists():
        target_dir = boilerplate_dir / "deepsculpt_legacy"
        if target_dir.exists():
            print("  ⚠️  Target already exists, removing old version")
            shutil.rmtree(target_dir)
        shutil.move(str(legacy_dir), str(target_dir))
        print("  ✅ Moved deepsculpt_legacy to boilerplate/")
    else:
        print("  ⚠️  deepsculpt_legacy not found, skipping")
    
    print()
    
    # Step 2: Rename deepsculpt_v2 to deepsculpt
    print("📝 Step 2: Renaming deepsculpt_v2 to deepsculpt")
    v2_dir = project_root / "deepsculpt_v2"
    new_dir = project_root / "deepsculpt"
    
    if not v2_dir.exists():
        print("  ❌ deepsculpt_v2 not found!")
        return
    
    if new_dir.exists():
        print("  ⚠️  deepsculpt already exists, removing old version")
        shutil.rmtree(new_dir)
    
    shutil.move(str(v2_dir), str(new_dir))
    print("  ✅ Renamed to deepsculpt")
    
    print()
    
    # Step 3: Update imports in Python files
    print("🔧 Step 3: Updating imports in Python files...")
    
    python_files = list(new_dir.rglob("*.py"))
    updated_count = 0
    
    for py_file in python_files:
        # Update various import patterns
        changed = False
        changed |= update_file_content(py_file, "from deepsculpt_v2.", "from deepsculpt.")
        changed |= update_file_content(py_file, "import deepsculpt_v2", "import deepsculpt")
        changed |= update_file_content(py_file, "deepsculpt_v2/", "deepsculpt/")
        
        if changed:
            updated_count += 1
    
    print(f"  ✅ Updated {updated_count} Python files")
    
    print()
    
    # Step 4: Update notebooks
    print("📓 Step 4: Updating imports in notebooks...")
    
    notebook_files = list((new_dir / "notebooks").rglob("*.ipynb"))
    updated_count = 0
    
    for nb_file in notebook_files:
        changed = update_file_content(nb_file, "deepsculpt_v2", "deepsculpt")
        if changed:
            updated_count += 1
    
    print(f"  ✅ Updated {updated_count} notebook files")
    
    print()
    
    # Step 5: Update scripts
    print("📜 Step 5: Updating path references in scripts...")
    
    script_files = list((new_dir / "scripts").rglob("*.py"))
    updated_count = 0
    
    for script_file in script_files:
        changed = update_file_content(script_file, "deepsculpt_v2", "deepsculpt")
        if changed:
            updated_count += 1
    
    print(f"  ✅ Updated {updated_count} script files")
    
    print()
    print("🎉 Restructuring complete!")
    print()
    print("Summary:")
    print("  ✅ Moved deepsculpt_legacy → boilerplate/deepsculpt_legacy")
    print("  ✅ Renamed deepsculpt_v2 → deepsculpt")
    print("  ✅ Updated all imports and paths")
    print()
    print("Next steps:")
    print("  1. Test the imports: python -c 'import deepsculpt'")
    print("  2. Run tests if available")
    print("  3. Commit changes:")
    print("     git add -A")
    print("     git commit -m 'Restructure: move legacy to boilerplate, rename v2 to main'")


if __name__ == "__main__":
    main()
