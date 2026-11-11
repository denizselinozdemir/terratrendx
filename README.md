# TerraTrendX: Turkey's long-term temperature trend

NASA Space Apps Challenge 2026, challenge "Be an Earth System Trend Detective".

**Live app:** https://terratrendx.streamlit.app/

TerraTrendX is an interactive Streamlit app that measures how Turkey's annual mean
2 m air temperature has changed since 1950, how large and how certain that change is,
and whether it differs between Turkey's seven geographic regions. Every statistic in the
app comes with a plain-language explanation and its limitations.

*(Türkçe özet: Uygulama Türkiye'nin 1950–2025 yıllık ortalama 2 metre hava sıcaklığındaki
değişimi, bu değişimin büyüklüğünü ve istatistiksel güvenilirliğini, yedi coğrafi bölge
karşılaştırmasını ve NASA MERRA-2 ile çapraz doğrulamayı gösterir. Arayüz Türkçedir.)*

## Research question

How has Turkey's annual mean 2 m air temperature changed between 1950 and 2025? How large
is the trend, is it statistically significant, and does it differ across regions?

## Data

| Dataset | Use |
|---|---|
| ERA5-Land monthly averaged data (Copernicus C3S), variable `t2m`, 0.1° grid, 1950–2025 | Main analysis |
| NASA MERRA-2 `M2TMNXSLV` (GES DISC), variable `T2M`, 0.5° x 0.625° grid, 1980–2025 | Independent cross-check |

Grid cells are selected with the centres inside the seven regional polygons
(`data/turkiye-regions.geojson`). Monthly values are converted to Celsius, area-averaged with
cosine-latitude weights, and aggregated to annual means weighted by the days in each month.

## Methods

- Linear trend with OLS and Newey-West (HAC) standard errors, with sensitivity to the lag choice.
- Mann-Kendall and Theil-Sen as non-parametric checks.
- Linear, quadratic and piecewise-linear models compared with AIC.
- Break year chosen by AIC; its uncertainty estimated with a circular moving-block bootstrap
  (1,000 replicates, block lengths 3, 5 and 7 years).
- Seven regional trends with Benjamini-Hochberg correction; 21 pairwise post-break slope
  comparisons with BH and Benjamini-Yekutieli corrections.

## Key results

- 1950–2025 national trend: **+0.209 °C per decade** (HAC 95% CI 0.107 to 0.311).
- Best piecewise model breaks at **1992** (bootstrap range about 1985–1999): slope about
  -0.12 °C per decade before, about +0.68 °C per decade after.
- All seven regions show a positive, significant trend (0.18 to 0.23 °C per decade).
- Cross-check, 1980–2025: ERA5-Land +0.53 vs MERRA-2 +0.44 °C per decade (confidence
  intervals overlap); correlation of detrended annual series 0.98.

## Limitations

- Both datasets are reanalyses, not station measurements, and they share much of their
  observational input. Early-period (1950s–1980s) trends are less constrained.
- The break year is chosen from the data, so p-values for post-break slopes and slope
  differences are descriptive, not formal tests.
- Slopes of the quadratic model at the ends of the period are not reliable.
- A statistically significant trend does not by itself identify a cause.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Repository contents

- `app.py`: the Streamlit app.
- `data/`: result tables (CSV), figures (PNG), region polygons and the MERRA-2 Turkey subset.
- `merra2_crosscheck.py`: downloads MERRA-2 via OPeNDAP (needs an Earthdata login) and compares
  it with ERA5-Land. Needs `earthaccess`, `xarray`, `geopandas` in addition to the app's packages.

## Data credits

Contains modified Copernicus Climate Change Service information (ERA5-Land) and NASA GMAO
MERRA-2 data distributed by the NASA GES DISC. See the dataset landing pages for the
official citation.