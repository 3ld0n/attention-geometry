#!/usr/bin/env bash
# Build PDF for "The Geometry Does Not Transmit"
# Run from the preprints/2026-07-21_generational_transmission/ directory.
# Requires: pandoc, pdflatex (BasicTeX or MacTeX)
#
# Usage:
#   bash build_pdf.sh
#
# Output:
#   the_geometry_does_not_transmit.pdf

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
  --output=the_geometry_does_not_transmit.pdf \
  --variable=papersize:letter

echo ""
echo "Done: the_geometry_does_not_transmit.pdf"
ls -lh the_geometry_does_not_transmit.pdf
