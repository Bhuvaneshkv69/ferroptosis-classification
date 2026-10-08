# CSV input format

The Analysis page takes **one** CSV format for every analysis type (classification, drug response, both).

## Structure

- One row per sample.
- One column per gene, named exactly as the gene symbol (case-sensitive), e.g. `GPX4`, `SLC7A11`, `ACSL4`.
- All 35 gene columns are required (listed below). Missing columns are reported by name.
- Extra columns (a sample ID, notes, …) are ignored. If the **first** column is not a gene it is used as the sample label.
- Values must be numeric. Blank or non-numeric cells in a required column are reported by column name.
- Negative values are accepted with a warning (clipped to 0 before the log2 transform).

## Scale

**Raw, untransformed expression** (RSEM-style, the scale TCGA's processed files use). The app applies
`log2(clip(x, 0) + 1)`, exactly as in training (`src/preprocessing.py::build_tcga_dataset`).
If your values are already log2-transformed, tick **Input options → My values are already log2-transformed**.

Download the exact template from the Analysis page ("Download CSV template").

```
ABCC1,ACSL4,AIFM2,AKR1C1,...,ZFP69B
1520.4,880.2,410.7,275.9,...,63.1
```
(placeholder values for illustration only)

## Drug response from the same file

The drug-response model was trained on cell-line **microarray** (GDSC, log2 RMA) values, a different scale from
tumour RNA-seq. When a tumour sample is sent to the drug-response model the app maps each gene onto the cell-line
scale by quantile matching against the two cohorts (`src/predict.py::harmonize_to_gdsc`). This step is
**exploratory and has not been validated against paired measurements**; the app says so next to the result.
To use the drug model on its native scale, choose **Reference sample → GDSC cell line**.

The compound is chosen in the app from the GDSC compound panel; it is not part of the CSV.

## Multiple samples in one file

All rows are scored and shown in a table; pick one row at a time for the explanation and drug-response panels.

## Required gene panel

```
ABCC1, ACSL4, AIFM2, AKR1C1, ALOX12, ATF3, ATG5, ATG7, ATP6V1G2, BID,
CARS1, CDKN2A, DRD4, FDFT1, GDF15, GPX4, HAMP, HOXC11, LPCAT3, MMP7,
NCOA4, NFS1, NOS2, NOX4, PCOLCE2, RRM2, SLC2A3, SLC39A8, SLC7A11,
TFAP2C, TXNIP, VLDLR, WIPI1, YAP1, ZFP69B
```

Provenance for each gene is in `data/unified_gene_signature.csv` (browsable on the Data page).

## No CSV?

**Enter sample** gives one numeric field per gene, pre-filled with the training cohort's median and annotated with
its typical range. **Reference sample** loads a real TCGA tumour or GDSC cell line.

## Training tables (Train / Compare page)

A custom training table needs `sampleId`, `patientId` (unique — one row per patient), `msi_label`
(`MSI-H` or `MSS/MSI-L`) and the 35 gene columns on the **log2(x+1)** scale. It is checked by
`src/validation.py::validate_training_dataframe` (≥ 150 labelled samples, ≥ 25 MSI-H, no repeated patients).
