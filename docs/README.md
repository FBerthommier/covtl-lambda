# covtl-lambda-patch — Documentation

Reference manual of the lambda-morph option (English US), LaTeX
source + compiled PDF, plus the cover source and the figure script.

| file | content |
|---|---|
| `manual.pdf` | Reference manual (14 pages): motivation and provenance, the morphing model (v1.2.0 target semantic on the tautosyllabic CC cluster + the blend option: interpolation rule, frame-alignment contract, glottis policy, guarantees), patch procedure (apply/revert internals), CLI reference, Python API reference, the worked assimilation continuum with figure walkthroughs, test inventory (53), limitations and extensions, references. |
| `manual.tex` | LaTeX source of the manual body (starts at the table of contents; the cover is generated separately and merged). |
| `cover.html` / `cover.pdf` | Cover page (HTML source + rendered vector PDF). |
| `covtl_lambda_C1C2_report.pdf` / `.tex` | Theoretical synthesis (English): the nature of the λ parameter (polar targets of the cluster branch) and C2 invariance. |
| `rapport_lambda_C1C2.pdf` / `.tex` | Same theoretical synthesis, French. |
| `figures/fig_continuum_data.png` | Data figure of §6: JA/TTX time series across the P1 target continuum + articulatory ground truth vs lambda. |
| `figures/make_figures.py` | Regenerates the figure from the output of `examples/lambda_continuum.py`. |

## Building the manual

The body is compiled with XeTeX/Tectonic (Liberation Serif/Mono,
Times-metric, full IPA coverage); the cover is rendered from HTML via
Playwright and merged as page 0:

```bash
# body (TOC first, references resolved in 2 passes)
tectonic manual.tex && tectonic manual.tex
# or: python <pdf-skill>/scripts/pdf.py convert.latex manual.tex --runs 2

# cover (single-page vector PDF)
node <pdf-skill>/scripts/html2poster.js cover.html --output cover.pdf --width 794px

# merge (strict A4 normalization)
python - <<'EOF'
from pypdf import PdfReader, PdfWriter
A4 = (595.28, 841.89)
def norm(p):
    if abs(float(p.mediabox.width)-A4[0]) > 0.1 or abs(float(p.mediabox.height)-A4[1]) > 0.1:
        p.scale_to(*A4)
    return p
w = PdfWriter()
w.add_page(norm(PdfReader('cover.pdf').pages[0]))
for p in PdfReader('manual.pdf').pages: w.add_page(norm(p))
with open('manual_merged.pdf','wb') as f: w.write(f)
EOF
```

Under pdfLaTeX, replace the `\setmainfont`/`\setmonofont` lines of
`manual.tex` with `\usepackage{lmodern}` (IPA coverage should then be
checked).

## Regenerating the figure

```bash
# 1. run the example (patched pipeline checkout, from its root)
python /path/to/covtl-lambda-patch/examples/lambda_continuum.py --out lambda_out

# 2. render the figure from those outputs
python figures/make_figures.py --data lambda_out
```
