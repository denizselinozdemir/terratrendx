"""NASA MERRA-2 ile Türkiye ve 7 bölge için yıllık sıcaklık ve yağış serileri + eğilim istatistikleri.

Önce veriyi indir (Earthdata girişi gerekir):
    python merra2_crosscheck.py indir          # sıcaklık (M2TMNXSLV, T2M)  -> data/merra2_t2m_turkiye.npz
    python merra2_crosscheck.py indir_yagis    # yağış (M2TMNXFLX, PRECTOTCORR) -> data/merra2_prectotcorr_turkiye.npz
Sonra:
    python merra2_tam.py

Üretilen dosyalar (data/, UTF-8-SIG):
    merra2_sicaklik_turkiye_yillik.csv, merra2_sicaklik_bolgesel_yillik.csv,
    merra2_sicaklik_ulusal_egilim.csv,  merra2_sicaklik_bolgesel_egilim.csv
    merra2_yagis_turkiye_yillik.csv,    merra2_yagis_bolgesel_yillik.csv,
    merra2_yagis_ulusal_egilim.csv,     merra2_yagis_bolgesel_egilim.csv
Dönem: 1980-2025 (MERRA-2'nin başladığı yıldan itibaren).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import yagis_isle as yi

DATA_DIR = Path(__file__).parent / "data"
START, END = 1980, 2025
DOSYALAR = {
    "sicaklik": ("merra2_t2m_turkiye.npz", "T2M"),
    "yagis": ("merra2_prectotcorr_turkiye.npz", "PRECTOTCORR"),
}
SICAKLIK_ARALIK = (3.0, 16.0)     # Türkiye ortalaması yaklaşık 8-13 °C
YAGIS_ARALIK = (350.0, 900.0)     # mm/yıl


def npz_ac(ad, anahtar):
    for p in (DATA_DIR / ad, Path.home() / ad):
        if p.exists():
            with open(p, "rb") as f:
                z = np.load(f)
                return (np.asarray(z[anahtar], dtype=float), pd.to_datetime(z["time"]),
                        np.asarray(z["lat"], dtype=float), np.asarray(z["lon"], dtype=float))
    return None


def yillik_ortalama(ay_degerleri, zaman):
    """Aylık değerler -> gün ağırlıklı yıllık ortalama; yalnızca 12 ayı tam yıllar."""
    idx = pd.DatetimeIndex(zaman)
    df = pd.DataFrame({"v": np.asarray(ay_degerleri, dtype=float), "g": idx.days_in_month.values.astype(float),
                       "yil": idx.year})
    df = df.dropna()
    sayi = df.groupby("yil")["v"].count()
    tam = sayi[sayi == 12].index
    df = df[df["yil"].isin(tam)]
    out = df.groupby("yil").apply(lambda g: np.average(g["v"], weights=g["g"]))
    return out


def yillik_toplam_yagis(mm_gun, zaman):
    """Aylık ortalama mm/gün -> yıllık toplam mm (ay uzunluğuyla çarp, topla); 12 ayı tam yıllar."""
    idx = pd.DatetimeIndex(zaman)
    mm_ay = np.asarray(mm_gun, dtype=float) * idx.days_in_month.values
    df = pd.DataFrame({"v": mm_ay, "yil": idx.year}).dropna()
    sayi = df.groupby("yil")["v"].count()
    toplam = df.groupby("yil")["v"].sum()
    return toplam[sayi == 12]


def egim_istatistikleri(seri):
    """yagis_isle.egim_istatistikleri ile aynı hesap; sütun adları birimden bağımsız."""
    r = yi.egim_istatistikleri(seri)
    return {
        "ortalama": r["ortalama_mm"], "hac_egim_on_yil": r["hac_egim_mm_on_yil"],
        "hac_ga_alt": r["hac_ga_alt"], "hac_ga_ust": r["hac_ga_ust"], "hac_p": r["hac_p"],
        "yuzde_on_yil": r["yuzde_on_yil"], "kendall_tau": r["kendall_tau"], "kendall_p": r["kendall_p"],
        "sen_egim_on_yil": r["sen_egim_mm_on_yil"], "sen_ga_alt": r["sen_ga_alt"], "sen_ga_ust": r["sen_ga_ust"],
        "yil_baslangic": r["yil_baslangic"], "yil_bitis": r["yil_bitis"],
    }


def kes(s):
    return s[(s.index >= START) & (s.index <= END)]


def degisken_isle(ad, vals, zaman, lat, lon, ulusal_m, bolgesel_m):
    deger_adi = "sicaklik_C" if ad == "sicaklik" else "yagis_mm"

    def seri_yap(maske):
        ay = yi.ay_serisi(vals, lat, maske)
        if ad == "sicaklik":
            return kes(yillik_ortalama(ay - 273.15, zaman))
        return kes(yillik_toplam_yagis(ay * 86400.0, zaman))   # kg m-2 s-1 -> mm/gün

    ulusal = seri_yap(ulusal_m)
    ort = float(ulusal.mean())
    lo, hi = SICAKLIK_ARALIK if ad == "sicaklik" else YAGIS_ARALIK
    print(f"[{ad}] ulusal ortalama {ort:.2f} | {len(ulusal)} yıl ({ulusal.index.min()}-{ulusal.index.max()})")
    if not (lo <= ort <= hi):
        raise SystemExit(f"DURDUM: {ad} ortalaması {ort:.2f}, beklenen {lo}-{hi} dışında. Bu satırı bana yapıştır.")

    pre = f"merra2_{ad}_"
    ulusal.rename(deger_adi).rename_axis("yil").reset_index().to_csv(
        DATA_DIR / (pre + "turkiye_yillik.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame([egim_istatistikleri(ulusal)]).to_csv(
        DATA_DIR / (pre + "ulusal_egilim.csv"), index=False, encoding="utf-8-sig")

    yillik, egilim = [], []
    for b, m in sorted(bolgesel_m.items()):
        s = seri_yap(m)
        print(f"  {b}: {int(m.sum())} hücre, ortalama {s.mean():.2f}")
        yillik.append(pd.DataFrame({"bolge": b, "yil": s.index.values, deger_adi: s.values}))
        egilim.append({"bolge": b, **egim_istatistikleri(s)})
    bol = pd.DataFrame(egilim)
    bol["BH_duzeltilmis_p"] = yi.bh(bol["hac_p"].values)
    bol["anlamli_mi"] = bol["BH_duzeltilmis_p"] < 0.05
    pd.concat(yillik).to_csv(DATA_DIR / (pre + "bolgesel_yillik.csv"), index=False, encoding="utf-8-sig")
    bol.to_csv(DATA_DIR / (pre + "bolgesel_egilim.csv"), index=False, encoding="utf-8-sig")

    u = egim_istatistikleri(ulusal)
    birim = "°C" if ad == "sicaklik" else "mm"
    print(f"  ULUSAL eğim: {u['hac_egim_on_yil']:+.3f} {birim}/10 yıl (GA {u['hac_ga_alt']:+.3f} / {u['hac_ga_ust']:+.3f}), "
          f"p={u['hac_p']:.3g}, Kendall p={u['kendall_p']:.3g}")
    print(bol[["bolge", "ortalama", "hac_egim_on_yil", "BH_duzeltilmis_p", "anlamli_mi"]].round(3).to_string(index=False))


def main():
    ilk = None
    for ad, (dosya, anahtar) in DOSYALAR.items():
        if npz_ac(dosya, anahtar) is not None:
            ilk = (ad, dosya, anahtar)
            break
    if ilk is None:
        raise SystemExit("MERRA-2 .npz dosyası bulunamadı. Önce 'python merra2_crosscheck.py indir' çalıştır.")

    bolge_adlari = yi.VARSAYILAN_BOLGELER
    if yi.SICAKLIK_BOLGE_CSV.exists():
        adlar = pd.read_csv(yi.SICAKLIK_BOLGE_CSV, encoding="utf-8-sig")
        kol = next((c for c in ("bolge", "BOLGE_ADI") if c in adlar.columns), None)
        if kol:
            bolge_adlari = sorted(adlar[kol].dropna().unique().tolist())

    DATA_DIR.mkdir(exist_ok=True)
    yapilan = 0
    for ad, (dosya, anahtar) in DOSYALAR.items():
        sonuc = npz_ac(dosya, anahtar)
        if sonuc is None:
            print(f"[{ad}] {dosya} yok, atlandı.")
            continue
        vals, zaman, lat, lon = sonuc
        ulusal_m, bolgesel_m = yi.maskeleri_olustur(lat, lon, bolge_adlari)
        print(f"[{ad}] ulusal maske hücre sayısı: {int(ulusal_m.sum())}")
        degisken_isle(ad, vals, zaman, lat, lon, ulusal_m, bolgesel_m)
        yapilan += 1
    print(f"\nTamam: {yapilan} değişken işlendi. Dosyalar: data/merra2_*.csv")


if __name__ == "__main__":
    main()