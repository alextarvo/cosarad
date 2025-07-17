#!/bin/bash

# This file renames the desk_... GT files in th eAnomaly-ShapeNet dataset, so they
# will have the consistent naming scheme desk0_...

for file in desk_*.txt; do
    if [[ "$file" =~ desk_([a-z]+[0-9])\.txt ]]; then
        rest="${BASH_REMATCH[1]}"
        newname="desk0_${rest}.txt"
        echo "Renaming: $file → $newname"
        mv "$file" "$newname"
    fi
done