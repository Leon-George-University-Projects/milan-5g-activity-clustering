# Milan 5G Activity Clustering

Spatial and temporal clustering of telecommunications activity from the
[Milan Telecom Dataset](https://doi.org/10.7910/DVN/EGZHFV). The project groups
city grid cells by network usage patterns and compares K-Means with Ward
hierarchical clustering.

## Authors

This project was developed collaboratively by:

- [Leonidas Gourvelos](https://github.com/LeonGour)
- [Giorgos Lalas](https://github.com/GEOLAL03)

It was completed for the University of Patras course **Next Generation Network
Architectures, Technologies and Applications** (Project Topic 13, 2026).

## Project overview

The analysis processes telecom activity recorded across Milan during November
and December 2013. It combines SMS, call and internet activity with temporal
and spatial features to identify urban zones with similar network behaviour.

The pipeline includes:

- memory-efficient loading and aggregation of the raw daily files;
- grouping of the 100x100 Milan grid into super-cells;
- activity, peak-hour, night and weekend feature engineering;
- median imputation and standardisation;
- K-Means and Ward hierarchical clustering;
- PCA, dendrograms and internal cluster-quality metrics;
- temporal profiles and spatial cluster maps;
- an extended experiment with explicit spatial-distance features.

## Main findings

The baseline analysis uses five clusters. Ward clustering achieved a slightly
higher Silhouette score and a lower Davies-Bouldin score, while K-Means achieved
the higher Calinski-Harabasz score and produced a clearer geometric separation
in the PCA projection.

| Method | Silhouette | Davies-Bouldin | Calinski-Harabasz |
| --- | ---: | ---: | ---: |
| K-Means | 0.304 | 1.098 | 977.5 |
| Hierarchical (Ward) | 0.331 | 1.081 | 860.2 |

The clusters distinguish low-activity peripheral areas, mixed urban zones and
the high-activity commercial centre. The resulting profiles can support network
planning, resource allocation and network-slicing studies.

## Repository structure

```text
src/
  clustering_analysis.py                  # baseline analysis
  clustering_analysis_spatial.py          # extension with spatial features
docs/
  report.pdf                              # full report in Greek
  presentation.pptx                       # project presentation in Greek
data/
  README.md                               # dataset instructions
requirements.txt
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset

The raw dataset is not included because of its size. Download the Milan Telecom
Dataset from Harvard Dataverse and place the daily files matching
`sms-call-internet-mi-*.txt` inside `data/`.

Dataset DOI: [10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV)

## Run the analysis

From the repository root:

```powershell
python src/clustering_analysis.py
```

To run the experiment that adds spatial-distance features:

```powershell
python src/clustering_analysis_spatial.py
```

The scripts save the generated figures in the current working directory. The
full analysis, interpretation and references are available in the report.

## Technologies

Python, pandas, NumPy, SciPy, scikit-learn and Matplotlib.
