#!/usr/bin/env bash
# Build PDF for "Latent Iteration as Renormalization"
# Run from the preprints/2026-07-20_latent_rg_flow/ directory.
# Requires: pandoc, pdflatex (BasicTeX or MacTeX)
#
# Usage:
#   bash build_pdf.sh
#
# Output:
#   latent_iteration_as_renormalization.pdf

set -euo pipefail
cd "$(dirname "$0")"

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
  --output=latent_iteration_as_renormalization.pdf \
  --variable=papersize:letter

echo ""
echo "Done: latent_iteration_as_renormalization.pdf"
ls -lh latent_iteration_as_renormalization.pdf
