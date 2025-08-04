#!/usr/bin/env python3
"""
Sync dependencies between pyproject.toml and requirements.txt

This script helps maintain both files in sync for projects that want to support
both traditional pip workflows and modern Python packaging.

Usage:
    python sync_deps.py [--from-requirements | --from-pyproject]
    
    --from-requirements: Update pyproject.toml from requirements.txt (default)
    --from-pyproject:    Update requirements.txt from pyproject.toml
    --help:             Show this help message
"""

import tomllib
import sys
import re
from pathlib import Path

def extract_deps_from_pyproject():
    """Extract dependencies from pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        sys.exit(1)
    
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    return data.get("project", {}).get("dependencies", [])

def extract_deps_from_requirements():
    """Extract dependencies from requirements.txt"""
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        print("Error: requirements.txt not found")
        sys.exit(1)
    
    dependencies = []
    with open(requirements_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith("#"):
                dependencies.append(line)
    
    return dependencies

def update_requirements_txt(dependencies):
    """Update requirements.txt with dependencies from pyproject.toml"""
    requirements_path = Path("requirements.txt")
    
    # Sort dependencies for consistency
    sorted_deps = sorted(dependencies)
    
    with open(requirements_path, "w") as f:
        f.write("# This file is auto-generated from pyproject.toml\n")
        f.write("# To update, run: python sync_deps.py --from-pyproject\n")
        f.write("\n")
        for dep in sorted_deps:
            f.write(f"{dep}\n")
    
    print(f"Updated {requirements_path} with {len(dependencies)} dependencies")

def update_pyproject_toml(dependencies):
    """Update pyproject.toml with dependencies from requirements.txt"""
    pyproject_path = Path("pyproject.toml")
    
    # Read the current pyproject.toml
    with open(pyproject_path, "r") as f:
        content = f.read()
    
    # Sort dependencies for consistency
    sorted_deps = sorted(dependencies)
    
    # Create the new dependencies section
    deps_section = 'dependencies = [\n'
    for dep in sorted_deps:
        deps_section += f'    "{dep}",\n'
    deps_section += ']'
    
    # Replace the dependencies section using regex
    pattern = r'dependencies = \[[\s\S]*?\]'
    new_content = re.sub(pattern, deps_section, content, flags=re.MULTILINE)
    
    with open(pyproject_path, "w") as f:
        f.write(new_content)
    
    print(f"Updated {pyproject_path} with {len(dependencies)} dependencies")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print(__doc__)
        return
    
    # Default to syncing from requirements to pyproject
    from_requirements = True
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--from-pyproject":
            from_requirements = False
        elif sys.argv[1] == "--from-requirements":
            from_requirements = True
        else:
            print("Error: Unknown argument. Use --help for usage information.")
            sys.exit(1)
    
    if from_requirements:
        print("Syncing dependencies from requirements.txt to pyproject.toml...")
        dependencies = extract_deps_from_requirements()
        update_pyproject_toml(dependencies)
    else:
        print("Syncing dependencies from pyproject.toml to requirements.txt...")
        dependencies = extract_deps_from_pyproject()
        update_requirements_txt(dependencies)
    
    print("Sync complete!")

if __name__ == "__main__":
    main()
