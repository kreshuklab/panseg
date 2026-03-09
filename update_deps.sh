#!/bin/bash

conda update -n panseg-dev --all
conda export --from-history --no-builds -n panseg-dev -f environment-dev.yaml
# remove the prefix line
head -n -1 environment-dev.yaml >temp_env.yaml && mv temp_env.yaml environment-dev.yaml

cat >>environment-dev.yaml <<'EOF'
  - pip:
      - markdown-exec
      - -e .
EOF
