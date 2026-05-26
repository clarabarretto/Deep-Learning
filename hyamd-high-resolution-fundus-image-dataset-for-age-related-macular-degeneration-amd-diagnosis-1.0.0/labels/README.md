# HYAMD: Hillel Yaffe Age-Related Macular Degeneration Dataset

The **HYAMD** dataset is a curated collection of **1,560 high-resolution Digital Fundus Images (DFIs)** from **325 patients** collected at the **Hillel Yaffe Medical Center**, Hadera, Israel (Helsinki approval number: 0029-24-HYMC). It provides **longitudinal data from 2021 to 2024** for the study and identification of **Age-Related Macular Degeneration (AMD)** using machine learning models.

This is, to the best of our knowledge, the **first open-access retinal dataset from an Israeli cohort**.

## Dataset Contents

- `images/` — Folder containing all DFIs in `.png` format. Each image has a resolution of **1960 × 1934 pixels** and was acquired using a **Topcon DRI OCT Triton** camera (45° field of view).
- `labels.csv` — Metadata file with labels and patient-level information.

### `labels.csv` columns:
| Column      | Description                                                        |
|-------------|--------------------------------------------------------------------|
| `image_id`  | Filename of the image (matches name in `images/`)                  |
| `patient_id`| Unique identifier for the patient                                  |
| `sex`       | `M` or `F`                                                         |
| `age`       | Age at the time of the visit                                       |
| `side`      | Eye side: `R` (right) or `L` (left)                                |
| `AMD`       | AMD label (see label definitions below)                            |

### Example row:
981616684_R,981616684,F,78,R,2
## Label Definitions

| Label | Description                                     |
|-------|-------------------------------------------------|
| 0     | Control group (diabetic retinopathy without AMD)|
| 1     | Early AMD                                       |
| 2     | Intermediate-to-late AMD                        |

> The AMD labels were determined via comprehensive clinical ophthalmic evaluation, supported by Optical Coherence Tomography (OCT), including OCT angiography. DR cases were clinically diagnosed using macular OCT, fluorescein angiography (when relevant), and widefield imaging.

## Intended Use

This dataset was developed to support **automated AMD identification from single DFIs** using machine learning. Annotations are **gold-standard**, based on full ophthalmic evaluation, not solely on fundus image interpretation.

## Citation

If you use this dataset, please cite:

**Meisel Meishar, Cohen Benjamin A., Baskin Meital, Behar Joachim A., Berkowitz Eran.**  
*HYAMD: High-Resolution Fundus Image Dataset for Age-Related Macular Degeneration (AMD) Diagnosis*. 2025.

## License and Contact

This dataset is released for **academic, non-commercial use only**. For inquiries, please contact the authors.

---
