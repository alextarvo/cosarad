#!/bin/bash

# This file renames the desk_... mesh files in th eAnomaly-ShapeNet dataset, so they
# will have the consistent naming scheme desk0_...


for file in desk_*.obj; do
    if [[ "$file" =~ desk_([a-z]+[0-9])\.obj ]]; then
        rest="${BASH_REMATCH[1]}"
        newname="desk0_${rest}.obj"
        echo "Renaming: $file → $newname"
        mv "$file" "$newname"
    fi
done