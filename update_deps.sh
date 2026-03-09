#!/bin/bash

# This script updates the panseg-dev conda environment and its yaml file.
# Only pinned packages stay pinned.
# The `environment.yaml` file gets updated from the dev environment.
# Both files can contain non-pinned packages.
# Please don't forget to update the conda-recipe/meta.yaml

ENV_FILE="environment.yaml"
DEV_ENV_FILE="environment-dev.yaml"

echo "Updating panseg-dev"
conda update -n panseg-dev --all || exit
echo "Updated panseg-dev, exporting.."
conda export --from-history --no-builds -n panseg-dev -f $DEV_ENV_FILE || exit
# remove the prefix line
head -n -1 $DEV_ENV_FILE >temp_env.yaml && mv temp_env.yaml $DEV_ENV_FILE

cat >>environment-dev.yaml <<'EOF'
  - pip:
      - markdown-exec
      - -e .
EOF

TEMP_ENV="tmp"

# Create associative arrays to store package information
declare -A dev_packages

# Extract packages from environment-dev.yaml into associative arrays
echo "DEBUG: Reading packages from $DEV_ENV_FILE" >&2
while IFS= read -r line; do
  # Check if we've reached the pip section
  if [[ "$line" =~ ^[[:space:]]*-[[:space:]]pip: ]]; then
    echo "DEBUG: Reached pip section in dev environment" >&2
    break
  fi

  # Check if we're in the dependencies section
  if [[ "$line" =~ ^[[:space:]]*dependencies:[[:space:]]*$ ]]; then
    echo "DEBUG: Entering dependencies section in dev environment" >&2
    in_deps=true
    continue
  fi

  # Skip lines that aren't in dependencies or are pip section
  if [[ ! "$line" =~ ^[[:space:]]*- ]] || [[ ! "${in_deps:-false}" == true ]]; then
    continue
  fi

  # Extract package name from line
  package_line="${line#*-[[:space:] ]}"
  package_line="${package_line%%[[:space:]]*}"

  # Extract just the package name part (before =, <, >)
  pkg_name=""
  if [[ "$package_line" =~ ^([^=<>]+) ]]; then
    pkg_name="${BASH_REMATCH[1]}"
  fi

  # Only process if we have a valid package name
  if [[ -n "$pkg_name" ]]; then
    # Store the full package line
    dev_packages["$pkg_name"]="$line"
    echo "DEBUG: Stored package $pkg_name from dev environment" >&2
  fi
done <"$DEV_ENV_FILE"

# Process environment.yaml and update matching packages
echo "DEBUG: Processing main environment file $ENV_FILE" >&2
{
  in_deps=false
  in_pip=false

  while IFS= read -r line; do
    # Check if we've reached the pip section
    if [[ "$line" =~ ^[[:space:]]*-[[:space:]]pip: ]]; then
      echo "DEBUG: Reached pip section in main environment" >&2
      in_pip=true
      echo "$line"
      continue
    fi

    # Check if we're entering dependencies section
    if [[ "$line" =~ ^[[:space:]]*dependencies:[[:space:]]*$ ]]; then
      echo "DEBUG: Entering dependencies section in main environment" >&2
      in_deps=true
      echo "$line"
      continue
    fi

    # If we're in pip section, just print the line
    if [[ "${in_pip:-false}" == true ]]; then
      echo "$line"
      continue
    fi

    # Check if we're in dependencies and have a package line
    if [[ "${in_deps:-false}" == true ]] && [[ "$line" =~ ^[[:space:]]*- ]]; then
      # Extract package name from line
      package_line="${line#*-[[:space:] ]}"
      package_line="${package_line%%[[:space:]]*}"

      # Extract just the package name part (before =, <, >)
      pkg_name=""
      if [[ "$package_line" =~ ^([^=<>]+) ]]; then
        pkg_name="${BASH_REMATCH[1]}"
      fi

      # If this matches a package from environment-dev.yaml, use the environment-dev.yaml version
      if [[ -n "$pkg_name" ]] && [[ -n "${dev_packages[$pkg_name]}" ]]; then
        echo "DEBUG: Updating package $pkg_name from dev version: ${dev_packages[$pkg_name]}" >&2
        echo "${dev_packages[$pkg_name]}"
      else
        echo "$line"
      fi
    else
      echo "$line"
    fi
  done <"$ENV_FILE"
} >"$TEMP_ENV"

# Replace original file
echo "DEBUG: Replacing original file with updated content" >&2
mv "$TEMP_ENV" "$ENV_FILE"

echo "Sync complete!"
