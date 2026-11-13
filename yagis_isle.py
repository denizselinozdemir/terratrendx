"""ERA5-Land aylık toplam yağıştan Türkiye ve bölgesel yıllık yağış serileri + eğilim istatistikleri.

Kullanım (EarthScope_v2 klasöründe):
    python yagis_isle.py                      # data/yagis.nc dosyasını okur
    python yagis_isle.py data/baska_ad.nc     # farklı bir dosya adı
    python yagis_isle.py --aylik-toplam       # dosyadaki değerler zaten AYLIK TOPLAM ise (nadir)

Üretilen dosyalar (data/ içine, UTF-8-SIG):
    yagis_turkiye_yillik.csv, yagis_bolgesel_yillik.csv,
    yagis_ulusal_egilim.csv,  yagis_bolgesel_egilim.csv

Gereksinimler: pandas numpy scipy statsmodels xarray h5netcdf geopandas shapely
"""
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
GEOJSON_PATH = DATA_DIR / "turkiye-regions.geojson"
SICAKLIK_BOLGE_CSV = DATA_DIR / "bolgesel_yillik_sicaklik.csv"   # bölge adlarını buradan alırız
VARSAYILAN_BOLGELER = ["Akdeniz", "Doğu Anadolu", "Ege", "Güneydoğu Anadolu",
                       "Karadeniz", "Marmara", "İç Anadolu"]
START, END = 1950, 2025
BEKLENEN_ORT_MM = (350.0, 900.0)   # Türkiye yıllık ortalama yağışı yaklaşık 500-600 mm


# ----------------------------- yardımcılar -----------------------------

def norm(s):
    """Türkçe harfleri sadeleştirip karşılaştırılabilir hale getirir; 'Akdeniz Bölgesi' -> 'akdeniz'."""
    s = str(s).replace("İ", "i").replace("I", "ı").lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).replace("ı", "i").strip()
    return re.sub(r"\s+(bolgesi|bolgeleri|bolge|region)$", "", s)


def bh(p):
    """Benjamini–Hochberg düzeltilmiş p-değerleri."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def ay_serisi(vals, lat, maske):
    """(zaman, enlem, boylam) -> maske içindeki cos(enlem) ağırlıklı alan ortalaması (zaman,)."""
    w = np.cos(np.deg2rad(np.asarray(lat, dtype=float)))[:, None] * maske
    gecerli = np.isfinite(vals)
    wt = gecerli * w[None, :, :]
    top = wt.sum(axis=(1, 2))
    deger = (np.where(gecerli, vals, 0.0) * wt).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(top > 0, deger / top, np.nan)


def yillik_toplam(ay_degerleri, zaman, aylik_toplam=False):
    """Aylık değerler -> yıllık toplam yağış (mm). Girdi m/gün (günlük ortalama birikim) varsayılır.

    aylik_toplam=True ise girdi zaten aylık toplam (m) kabul edilir, gün sayısıyla çarpılmaz.
    Yalnızca 12 ayı tam olan yıllar döner.
    """
    idx = pd.DatetimeIndex(zaman)
    carpan = np.ones(len(idx)) if aylik_toplam else idx.days_in_month.values.astype(float)
    mm_ay = np.asarray(ay_degerleri, dtype=float) * 1000.0 * carpan
    df = pd.DataFrame({"v": mm_ay, "yil": idx.year})
    sayi = df.groupby("yil")["v"].count()
    toplam = df.groupby("yil")["v"].sum()
    toplam = toplam[sayi == 12]
    return toplam.rename("yagis_mm")


def hac_egim(y, yil, maxlags=3):
    import statsmodels.api as sm
    x = (np.asarray(yil, dtype=float) - np.mean(yil)) / 10.0
    m = sm.OLS(np.asarray(y, dtype=float), sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    ci = m.conf_int()
    return float(m.params[1]), float(ci[1, 0]), float(ci[1, 1]), float(m.pvalues[1])


def egim_istatistikleri(seri):
    """Yıl indeksli yıllık toplam serisi için HAC, Kendall ve Theil–Sen eğilimleri (birim: mm/10 yıl)."""
    from scipy import stats
    y = seri.values.astype(float)
    x = seri.index.values.astype(float)
    egim, lo, hi, p = hac_egim(y, x)
    ort = float(y.mean())
    tau, kp = stats.kendalltau(x, y)
    sen, _, sen_lo, sen_hi = stats.theilslopes(y, x, 0.95)
    return {
        "ortalama_mm": ort,
        "hac_egim_mm_on_yil": egim, "hac_ga_alt": lo, "hac_ga_ust": hi, "hac_p": p,
        "yuzde_on_yil": 100.0 * egim / ort,
        "kendall_tau": float(tau), "kendall_p": float(kp),
        "sen_egim_mm_on_yil": float(sen) * 10, "sen_ga_alt": float(sen_lo) * 10,
        "sen_ga_ust": float(sen_hi) * 10,
        "yil_baslangic": int(x.min()), "yil_bitis": int(x.max()),
    }


# ----------------------------- dosya okuma ve maske -----------------------------

def veri_ac(yol):
    """NetCDF'i Python'un kendi dosya açıcısıyla okur (Türkçe karakterli yollarda netCDF4 sorun çıkarır)."""
    import xarray as xr
    son_hata = None
    for engine in ("h5netcdf", "scipy"):
        try:
            with open(yol, "rb") as f:
                return xr.open_dataset(f, engine=engine).load()
        except Exception as e:   # biçim uyuşmazlığı vb.
            son_hata = e
    raise SystemExit(f"Dosya okunamadı: {yol}\nSon hata: {son_hata}")


def alani_hazirla(ds):
    """(değişken, zaman adı, enlem adı, boylam adı) -> (zaman, enlem, boylam) dizisi ve koordinatlar."""
    var = "tp" if "tp" in ds.data_vars else list(ds.data_vars)[0]
    tname = next((n for n in ("valid_time", "time") if n in ds.coords), None)
    latn = next((n for n in ("latitude", "lat") if n in ds.coords), None)
    lonn = next((n for n in ("longitude", "lon") if n in ds.coords), None)
    if None in (tname, latn, lonn):
        raise SystemExit(f"Koordinat adları tanınmadı: {list(ds.coords)}")
    da = ds[var]
    for extra in [d for d in da.dims if d not in (tname, latn, lonn)]:
        if da.sizes[extra] != 1:
            raise SystemExit(f"Beklenmeyen boyut '{extra}' ({da.sizes[extra]} değer). Dosya yapısını bana yapıştır.")
        da = da.isel({extra: 0})
    da = da.transpose(tname, latn, lonn)
    print(f"Değişken: {var} | birim: {da.attrs.get('units', '?')} | şekil: {da.shape}")
    return (da.values.astype(float), ds[tname].values, ds[latn].values, ds[lonn].values)


def bolge_eslesmesi(gdf, bolge_adlari):
    """GeoJSON satırlarını bölge adlarına eşler: {satır_indeksi: bölge_adı}. Bulamazsa None.

    Sayısal olmayan her sütun denenir (pandas sürümüne göre metin türü 'object' ya da 'str' olabilir).
    """
    hedef = {norm(b): b for b in bolge_adlari}
    for col in gdf.columns:
        if col == "geometry" or pd.api.types.is_numeric_dtype(gdf[col]):
            continue
        aday = {i: hedef[norm(v)] for i, v in gdf[col].items() if norm(v) in hedef}
        if len(set(aday.values())) == len(hedef):
            print(f"Bölge adı sütunu: '{col}'")
            return aday
    return None


def maskeleri_olustur(lat, lon, bolge_adlari):
    import geopandas as gpd
    import shapely
    gdf = gpd.read_file(GEOJSON_PATH).to_crs(4326)
    lon2, lat2 = np.meshgrid(lon, lat)

    def icinde(geom):
        try:
            return shapely.contains_xy(geom, lon2, lat2)
        except AttributeError:   # eski shapely
            return np.array([[geom.contains(shapely.geometry.Point(x, y)) for x in lon] for y in lat])

    birlesik = gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union
    ulusal = icinde(birlesik)

    eslesme = bolge_eslesmesi(gdf, bolge_adlari)
    if eslesme is None:
        raise SystemExit("GeoJSON'da 7 bölge adı eşleşmedi. Özellikler: "
                         f"{[c for c in gdf.columns if c != 'geometry']}\nİlk satırlar:\n{gdf.drop(columns='geometry').head(8)}")
    bolgesel = {}
    for i, ad in eslesme.items():
        m = icinde(gdf.geometry.loc[i])
        bolgesel[ad] = bolgesel.get(ad, np.zeros_like(m)) | m
    return ulusal, bolgesel


# ----------------------------- ana akış -----------------------------

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    aylik_toplam = "--aylik-toplam" in sys.argv
    yol = Path(args[0]) if args else DATA_DIR / "yagis.nc"
    if not yol.exists():
        raise SystemExit(f"Dosya bulunamadı: {yol}\nİndirdiğin yağış dosyasını data klasörüne 'yagis.nc' adıyla koy.")

    bolge_adlari = VARSAYILAN_BOLGELER
    if SICAKLIK_BOLGE_CSV.exists():
        adlar = pd.read_csv(SICAKLIK_BOLGE_CSV, encoding="utf-8-sig")
        kol = next((c for c in ("bolge", "BOLGE_ADI") if c in adlar.columns), None)
        if kol:
            bolge_adlari = sorted(adlar[kol].dropna().unique().tolist())
    print("Bölgeler:", bolge_adlari)

    ds = veri_ac(yol)
    vals, zaman, lat, lon = alani_hazirla(ds)
    ulusal_m, bolgesel_m = maskeleri_olustur(lat, lon, bolge_adlari)
    print("Maskedeki hücre sayısı (ulusal):", int(ulusal_m.sum()))

    ulusal = yillik_toplam(ay_serisi(vals, lat, ulusal_m), zaman, aylik_toplam)
    ulusal = ulusal[(ulusal.index >= START) & (ulusal.index <= END)]
    ort = float(ulusal.mean())
    print(f"Ulusal yıllık ortalama yağış: {ort:.0f} mm | yıl sayısı: {len(ulusal)} ({ulusal.index.min()}-{ulusal.index.max()})")
    if not (BEKLENEN_ORT_MM[0] <= ort <= BEKLENEN_ORT_MM[1]):
        raise SystemExit(
            f"DURDUM: ortalama {ort:.0f} mm, beklenen aralık {BEKLENEN_ORT_MM} dışında (Türkiye için yaklaşık 500-600 mm).\n"
            "Birim ya da biçim sorunu olabilir. Değer çok büyükse dosya zaten aylık toplam olabilir: "
            "'python yagis_isle.py --aylik-toplam' ile dene. Olmuyorsa bu satırı bana yapıştır.")

    DATA_DIR.mkdir(exist_ok=True)
    ulusal.rename_axis("yil").reset_index().to_csv(
        DATA_DIR / "yagis_turkiye_yillik.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([egim_istatistikleri(ulusal)]).to_csv(
        DATA_DIR / "yagis_ulusal_egilim.csv", index=False, encoding="utf-8-sig")

    yillik_satirlar, egilim_satirlar = [], []
    for ad, m in sorted(bolgesel_m.items()):
        s = yillik_toplam(ay_serisi(vals, lat, m), zaman, aylik_toplam)
        s = s[(s.index >= START) & (s.index <= END)]
        print(f"  {ad}: {int(m.sum())} hücre, ortalama {s.mean():.0f} mm/yıl")
        yillik_satirlar.append(pd.DataFrame({"bolge": ad, "yil": s.index.values, "yagis_mm": s.values}))
        egilim_satirlar.append({"bolge": ad, **egim_istatistikleri(s)})
    bol = pd.DataFrame(egilim_satirlar)
    bol["BH_duzeltilmis_p"] = bh(bol["hac_p"].values)
    bol["anlamli_mi"] = bol["BH_duzeltilmis_p"] < 0.05
    pd.concat(yillik_satirlar).to_csv(DATA_DIR / "yagis_bolgesel_yillik.csv", index=False, encoding="utf-8-sig")
    bol.to_csv(DATA_DIR / "yagis_bolgesel_egilim.csv", index=False, encoding="utf-8-sig")

    u = pd.read_csv(DATA_DIR / "yagis_ulusal_egilim.csv", encoding="utf-8-sig").iloc[0]
    print("\n=== ÖZET ===")
    print(f"Ulusal HAC eğimi: {u['hac_egim_mm_on_yil']:+.1f} mm/10 yıl (%95 GA {u['hac_ga_alt']:+.1f} / {u['hac_ga_ust']:+.1f}), "
          f"p={u['hac_p']:.3g}, yüzde: {u['yuzde_on_yil']:+.2f} %/10 yıl")
    print(f"Kendall tau={u['kendall_tau']:.3f} (p={u['kendall_p']:.3g}) | Theil–Sen: {u['sen_egim_mm_on_yil']:+.1f} mm/10 yıl")
    print(bol[["bolge", "ortalama_mm", "hac_egim_mm_on_yil", "yuzde_on_yil", "BH_duzeltilmis_p", "anlamli_mi"]].round(3).to_string(index=False))
    print("\nKaydedildi: data/yagis_*.csv")


if __name__ == "__main__":
    main()