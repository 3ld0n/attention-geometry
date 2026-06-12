#!/usr/bin/env bash
# Build PDF for "Attention on the Null Cone"
# Run from the preprints/2026-06-09_null_cone/ directory.
# Requires: pandoc, pdflatex (BasicTeX or MacTeX)
#
# Usage:
#   bash build_pdf.sh
#
# Output:
#   attention_on_the_null_cone.pdf

set -euo pipefail
cd "$(dirname "$0")"

# Add BasicTeX binaries if installed via Homebrew
if [ -d /Library/TeX/texbin ]; then
  export PATH="/Library/TeX/texbin:$PATH"
fi

echo "Checking dependencies..."
pandoc --version | head -1
pdflatex --version | head -1

echo ""
echo "Building PDF..."
pandoc manuscript.md \
  --bibliography=refs.bib \
  --citeproc \
  --pdf-engine=pdflatex \
  --output=attention_on_the_null_cone.pdf \
  --variable=papersize:letter

echo ""
echo "Done: attention_on_the_null_cone.pdf"
ls -lh attention_on_the_null_cone.pdf
