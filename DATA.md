# Data sources

No data is stored in this repository. All inputs are public and obtained from the
sources below. Paths in parentheses are where the pipeline expects the data
locally (all gitignored).

## ABIDE-I (site analysis)

- Preprocessed Connectomes Project, ABIDE Preprocessed Initiative.
- Pipeline C-PAC, strategy `filt_global`, derivative `rois_cc200` (Craddock CC200,
  200 regions), `.1D` region timeseries.
- Public S3, no credentials. URL template:
  `https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_cc200/<FILE_ID>_rois_cc200.1D`
- Phenotype: `.../ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv`
- `scripts/build_dataset.py` downloads, QC-filters (mean FD < 0.2, perc FD < 30,
  sites with N >= 30), and ComBat-GAM harmonizes to
  `data/processed/abide_harmonized.npz`. Analyzed cohort: 742 subjects, 14 sites.
- Docs: http://preprocessed-connectomes-project.org/abide/

## ADHD-200 (site replication, Athena pipeline)

- Neuro Bureau ADHD-200 Preprocessed, Athena pipeline (AFNI + FSL), CC200 region
  timeseries (`*_cc200_TCs.1D`), distributed as a single tarball
  `ADHD200_CC200_TCs_filtfix.tar` (all subjects, per-site phenotype + a KKI motion
  file). NOTE: this is a DIFFERENT engine (Athena, no global signal regression)
  from ABIDE C-PAC filt_global; the replication is conceptual, see
  `scripts/adhd200/README.md`.
- Available from NITRC (free account) and per the PCP ADHD-200 pages.
  Docs: http://preprocessed-connectomes-project.org/adhd200/
- Place the tarball at `data/adhd/ADHD200_CC200_TCs_filtfix.tar`; the build reads
  it directly (no extraction). Analyzed cohort after QC: 543 subjects, 5 sites.
- The 162-subject C-PAC benchmark on `fcp-indi` is too small (0 sites >= 30) and
  is NOT used.

## AOMIC PIOP2 (condition analysis)

- Amsterdam Open MRI Collection, PIOP2, OpenNeuro dataset `ds002790`, snapshot
  version 2.0.0 (the snapshot that includes derivatives).
- Public OpenNeuro S3, no registration. We use only the fmriprep derivatives, MNI
  space, for rest and working memory:
  `s3://openneuro.org/ds002790/derivatives/fmriprep/<sub>/func/<sub>_task-{restingstate,workingmemory}_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`
  plus the matching `_desc-confounds_regressors.tsv` and, for working memory, the
  raw events file `ds002790/<sub>/func/<sub>_task-workingmemory_acq-seq_events.tsv`.
- `scripts/condition_sms/fetch_parcellate.py` downloads these per subject,
  parcellates to Schaefer-200, deletes the BOLD, and saves small region
  timeseries to `data/condition_sms/ts/` (222 subjects with both runs). Rest is
  240 volumes, working memory 160 volumes, both acq-seq at TR 2.0 s.
- Dataset: https://openneuro.org/datasets/ds002790

## Atlases (fetched automatically by nilearn, cached locally)

- Schaefer 2018, 200 parcels, 7 networks, 2 mm (condition analysis).
- Harvard-Oxford (cort + sub, maxprob thr25 2mm) and Yeo 2011 7-network (anatomical
  labeling). ABIDE uses the PCP CC200 atlas `data/raw/cc200_roi_atlas.nii.gz`.

## ABIDE-I CCS derivatives (cross-pipeline robustness, Table S4)

- Same Preprocessed Connectomes Project source and the SAME retained subject set
  as the C-PAC site analysis above, but a DIFFERENT preprocessing pipeline (CCS,
  Connectome Computation System) so the only thing that changes is the pipeline.
- Pipeline `ccs`, strategy `filt_global`, derivative `rois_cc200`. Public S3, no
  credentials. URL template:
  `https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/ccs/filt_global/rois_cc200/<FILE_ID>_rois_cc200.1D`
- `scripts/cross_pipeline/download_ccs.py` downloads exactly the FILE_IDs already
  retained by the C-PAC analysis (read from `data/processed/abide_harmonized.npz`),
  so site membership, N and TR-per-site are identical by construction (no re-QC).
- CENTERING: the CCS `filt_global` `.1D` files retain the raw BOLD DC offset,
  whereas C-PAC `filt_global` `.1D` are demeaned. `scripts/cross_pipeline/
  run_ccs_site_diagnostic.py` therefore demeans each region per subject over the
  full series (`DEMEAN_TO_MATCH_CPAC`) so that the ONLY remaining difference from
  the C-PAC run is the pipeline. CCS also ships one fewer volume per subject, so
  the cohort-minimum crop lands at T=115 vs C-PAC's 116; all subjects/sites are
  retained. Both the as-shipped (confounded) and matched columns are reported.

## ABIDE-I Schaefer-200 build (region-selection sensitivity, Table S6)

- The same C-PAC ABIDE-I subjects, re-parcellated to the Schaefer-200 (7-network,
  2 mm) atlas so the site and condition analyses can be compared on one atlas.
- `scripts/site_schaefer/stream_parcellate.py` streams each subject's
  `filt_global` func_preproc NIfTI from the PCP S3 bucket
  (`.../Outputs/cpac/filt_global/func_preproc/<FILE_ID>_func_preproc.nii.gz`),
  parcellates to Schaefer-200 with `NiftiLabelsMasker`
  (`standardize="zscore_sample"`, `detrend=False`), and deletes each volume after
  use (peak disk = a few volumes, never the full download). Set the working
  directory with the `MERIDIAN_SCRATCH` environment variable (default
  `~/meridian_schaefer`).
- `scripts/site_schaefer/run_site_schaefer.py` then ComBat-GAM harmonizes to
  `data/processed/abide_schaefer_harmonized.npz`, consumed by
  `scripts/region_sensitivity.py`.
