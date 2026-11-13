"""MERRA-2 (NASA) ile ERA5-Land Türkiye sıcaklık serisinin çapraz kontrolü.

Üç adım, ayrı ayrı çalıştırılır:

  python merra2_crosscheck.py indir    # MERRA-2 aylık T2M'yi Türkiye kutusu için çek -> data/merra2_t2m_turkiye.nc
  python merra2_crosscheck.py analiz   # yıllık seriyi çıkar, ERA5-Land ile karşılaştır, CSV/PNG üret
  python merra2_crosscheck.py          # ikisini birden

Gereksinimler: pip install earthaccess xarray netCDF4 h5netcdf geopandas shapely
Earthdata hesabı gerekir (ücretsiz): https://urs.earthdata.nasa.gov
İlk 'indir' çalıştırmasında kullanıcı adı/şifre sorulur (earthaccess.login).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
NC_PATH = DATA_DIR / "merra2_t2m_turkiye.nc"
# data klasörü OneDrive içinde kilitliyse sonuç ev klasörüne yazılır; analiz ikisine de bakar.
NPZ_ADAYLAR = [DATA_DIR / "merra2_t2m_turkiye.npz", Path.home() / "merra2_t2m_turkiye.npz"]

# ERA5-Land analizinde kullandığın sınırlar (biraz pay bırakıldı)
LAT_MIN, LAT_MAX = 35.5, 42.5
LON_MIN, LON_MAX = 25.0, 45.5

START, END = 1980, 2025          # MERRA-2 1980'de başlar
ERA5_CSV = DATA_DIR / "turkiye_yillik_sicaklik.csv"

# Türkiye poligonu: ERA5-Land maskesinde kullandığın GeoJSON dosyasının adı.
# Bunu kendi dosya adınla değiştir. None bırakırsan maske yerine kutu ortalaması alınır.
GEOJSON_PATH = DATA_DIR / "turkiye-regions.geojson"


# ----------------------------- 1) İNDİRME -----------------------------

import re
import struct
import time
from concurrent.futures import ThreadPoolExecutor

# MERRA-2 ızgarası sabittir: enlem -90..90 (0.5° adım, 361 nokta), boylam -180..179.375 (0.625° adım, 576 nokta)
LAT0, DLAT = -90.0, 0.5
LON0, DLON = -180.0, 0.625
I_LAT0 = int(np.ceil((LAT_MIN - LAT0) / DLAT - 1e-9))
I_LAT1 = int(np.floor((LAT_MAX - LAT0) / DLAT + 1e-9))
I_LON0 = int(np.ceil((LON_MIN - LON0) / DLON - 1e-9))
I_LON1 = int(np.floor((LON_MAX - LON0) / DLON + 1e-9))
N_LAT, N_LON = I_LAT1 - I_LAT0 + 1, I_LON1 - I_LON0 + 1


def parse_dods(content, n_beklenen):
    """DAP2 .dods yanıtından ilk dizinin (T2M) float32 değerlerini okur."""
    marker = b"\nData:\n"
    pos = content.find(marker)
    if pos < 0:
        raise ValueError("Yanıtta 'Data:' bölümü yok. İlk 300 bayt: %r" % content[:300])
    pos += len(marker)
    n1, n2 = struct.unpack(">II", content[pos:pos + 8])
    if n1 != n2 or n1 != n_beklenen:
        raise ValueError("Beklenen %d değer, gelen %d" % (n_beklenen, n1))
    return np.frombuffer(content[pos + 8:pos + 8 + 4 * n1], dtype=">f4").astype(np.float32)


def opendap_url(granule):
    """Granülün OPeNDAP adresi: önce CMR kaydındaki bağlantı, yoksa veri bağlantısından türet."""
    for item in granule["umm"].get("RelatedUrls", []):
        u = item.get("URL", "")
        if "opendap" in u.lower() and u.endswith(".nc4"):
            return u
        if "opendap" in u.lower() and u.endswith(".nc4.html"):
            return u[:-5]
    for u in granule.data_links():
        if u.endswith(".nc4"):
            return re.sub(r"https://[^/]+/data/",
                          "https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/", u)
    raise ValueError("OPeNDAP adresi bulunamadı")


def tek_ay_cek(session, base_url, deneme=3, degisken="T2M"):
    ce = "%s[0:0][%d:%d][%d:%d]" % (degisken, I_LAT0, I_LAT1, I_LON0, I_LON1)
    url = base_url + ".dods?" + ce
    son_hata = None
    for k in range(deneme):
        try:
            r = session.get(url, timeout=120)
            if r.status_code in (401, 403):
                raise PermissionError(
                    "Erişim reddedildi (%d). Earthdata profilinde 'NASA GESDISC DATA ARCHIVE' "
                    "uygulaması onaylı mı?" % r.status_code)
            r.raise_for_status()
            return parse_dods(r.content, N_LAT * N_LON).reshape(N_LAT, N_LON)
        except PermissionError:
            raise
        except Exception as e:  # ağ hatası, zaman aşımı, ayrıştırma
            son_hata = e
            time.sleep(2 * (k + 1))
    raise RuntimeError("%s adresi alınamadı: %s" % (base_url, son_hata))


def indir_hizli(session=None, results=None, workers=12, short_name="M2TMNXSLV",
                degisken="T2M", cikti_ad=None):
    """OPeNDAP ile yalnızca Türkiye kutusunu iste (dosya başına yaklaşık 2 KB).

    Varsayılan: sıcaklık (M2TMNXSLV / T2M). Yağış için: short_name="M2TMNXFLX", degisken="PRECTOTCORR".
    """
    import earthaccess

    if cikti_ad is None:
        adaylar = NPZ_ADAYLAR
    else:
        adaylar = [DATA_DIR / cikti_ad, Path.home() / cikti_ad]
    if session is None:
        earthaccess.login()
        session = earthaccess.get_requests_https_session()
    if results is None:
        results = earthaccess.search_data(
            short_name=short_name, version="5.12.4",
            temporal=(f"{START}-01-01", f"{END}-12-31"))
    print(f"Bulunan dosya sayısı: {len(results)} (beklenen: {(END - START + 1) * 12})")
    if not results:
        raise SystemExit("Hiç dosya bulunamadı.")

    urls = [opendap_url(g) for g in results]
    print("Kutu: enlem %.1f-%.1f, boylam %.3f-%.3f, %d x %d hücre"
          % (LAT0 + I_LAT0 * DLAT, LAT0 + I_LAT1 * DLAT,
             LON0 + I_LON0 * DLON, LON0 + I_LON1 * DLON, N_LAT, N_LON))

    # Önce tek dosyayı seri dene: hata varsa hemen ve açık şekilde görürsün, oturum çerezi de oluşur.
    ilk = tek_ay_cek(session, urls[0], degisken=degisken)
    if degisken == "T2M":
        print("Deneme dosyası tamam. Ortalama T2M = %.2f K (beklenen yaklaşık 270-290)" % ilk.mean())
        makul = 230 < ilk.mean() < 320
    else:   # yağış akısı kg m-2 s-1: Türkiye için yaklaşık 2e-5 (1,7 mm/gün)
        print("Deneme dosyası tamam. Ortalama %s = %.3g kg m-2 s-1 = %.2f mm/gün (beklenen yaklaşık 0,5-4 mm/gün)"
              % (degisken, ilk.mean(), ilk.mean() * 86400))
        makul = 0 <= ilk.mean() < 5e-4
    if not makul:
        raise SystemExit("Deneme değeri mantıksız, durdum.")

    sonuc = {0: ilk}
    bitti = [1]

    def is_(i):
        arr = tek_ay_cek(session, urls[i], degisken=degisken)
        bitti[0] += 1
        if bitti[0] % 50 == 0:
            print(f"  {bitti[0]}/{len(urls)} tamam", flush=True)
        return i, arr

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, arr in ex.map(is_, range(1, len(urls))):
            sonuc[i] = arr

    zamanlar, veri = [], []
    for i, u in enumerate(urls):
        ym = re.search(r"\.(\d{6})\.nc4", u).group(1)
        zamanlar.append(pd.Timestamp(int(ym[:4]), int(ym[4:]), 1))
        veri.append(sonuc[i])
    order = np.argsort(zamanlar)
    lat = LAT0 + DLAT * np.arange(I_LAT0, I_LAT1 + 1)
    lon = LON0 + DLON * np.arange(I_LON0, I_LON1 + 1)
    veri_arr = np.stack([veri[k] for k in order])
    zaman_arr = np.array(zamanlar, dtype="datetime64[ns]")[order]
    DATA_DIR.mkdir(exist_ok=True)
    # netCDF4 Windows'ta Türkçe karakterli yollara (Ö, İ, Ş...) yazamaz ("Permission denied" der).
    # Bu yüzden sonuç numpy .npz olarak kaydedilir: Python dosyayı kendisi açar, yol sorunu olmaz.
    for p in adaylar:
        try:
            with open(p, "wb") as f:
                np.savez(f, **{degisken: veri_arr}, time=zaman_arr, lat=lat, lon=lon)
            print("Kaydedildi:", p, "| şekil:", veri_arr.shape)
            break
        except OSError as e:
            print("Yazılamadı:", p, "->", e)
            print("Sıradaki konum deneniyor...")
    else:
        raise SystemExit("Hiçbir konuma yazılamadı.")


def npz_oku():
    """Kaydedilen MERRA-2 verisini xarray DataArray olarak döndürür."""
    import xarray as xr
    for p in NPZ_ADAYLAR:
        if p.exists():
            with open(p, "rb") as f:
                z = np.load(f)
                return xr.DataArray(z["T2M"], dims=("time", "lat", "lon"),
                                    coords={"time": z["time"], "lat": z["lat"], "lon": z["lon"]},
                                    name="T2M")
    raise SystemExit("merra2_t2m_turkiye.npz bulunamadı: önce 'indir' adımını çalıştır.")


def indir_yavas():
    import earthaccess
    import xarray as xr

    earthaccess.login()  # Earthdata kullanıcı adı / şifre
    # M2TMNXSLV: MERRA-2 tavg1_2d_slv_Nx aylık ortalama (T2M burada)
    results = earthaccess.search_data(
        short_name="M2TMNXSLV",
        version="5.12.4",
        temporal=(f"{START}-01-01", f"{END}-12-31"),
    )
    print(f"Bulunan dosya sayısı: {len(results)} (beklenen: {(END - START + 1) * 12})")
    if not results:
        raise SystemExit("Hiç dosya bulunamadı: kısa ad/sürümü Earthdata Search'ten kontrol et.")

    files = earthaccess.open(results)  # akış (stream): tüm dosyaları diske indirmez
    ds = xr.open_mfdataset(files, engine="h5netcdf", combine="by_coords",
                           chunks={"time": 12})
    sub = ds["T2M"].sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
    DATA_DIR.mkdir(exist_ok=True)
    sub.load().to_netcdf(NC_PATH)
    print("Kaydedildi:", NC_PATH, "| şekil:", sub.shape)


# ----------------------------- 2) ANALİZ -----------------------------

def turkiye_maskesi(lat, lon):
    """Hücre merkezleri poligon içindeyse True. GeoJSON yoksa tüm kutu True."""
    if GEOJSON_PATH is None:
        print("UYARI: GeoJSON verilmedi, kutu ortalaması kullanılıyor (kıyıda deniz hücreleri girer).")
        return np.ones((len(lat), len(lon)), dtype=bool)
    import geopandas as gpd
    from shapely.geometry import Point
    union = gpd.read_file(GEOJSON_PATH).to_crs(4326).unary_union
    return np.array([[union.contains(Point(x, y)) for x in lon] for y in lat])


def yillik_seri(t2m_k):
    """Aylık T2M (Kelvin, DataArray) -> Türkiye yıllık ortalama (°C), gün ağırlıklı, cos(enlem) ağırlıklı."""
    mask = turkiye_maskesi(t2m_k["lat"].values, t2m_k["lon"].values)
    print("Maskedeki hücre sayısı:", int(mask.sum()))
    w_lat = np.cos(np.deg2rad(t2m_k["lat"].values))[:, None] * np.ones((1, len(t2m_k["lon"])))
    w = np.where(mask, w_lat, 0.0)

    vals = t2m_k.values - 273.15                       # (zaman, lat, lon)
    aylik = (vals * w).sum(axis=(1, 2)) / w.sum()      # alan ağırlıklı aylık ortalama
    idx = pd.to_datetime(t2m_k["time"].values)
    s = pd.Series(aylik, index=idx)
    gun = s.index.days_in_month
    df = pd.DataFrame({"t": s.values, "gun": gun, "yil": s.index.year})
    # Yalnızca 12 ayı tam olan yıllar
    tam = df.groupby("yil")["t"].transform("count") == 12
    df = df[tam]
    yillik = df.groupby("yil").apply(lambda g: np.average(g["t"], weights=g["gun"]))
    return yillik.rename("merra2_C")


def egim_hac(y, yil, maxlags=3):
    import statsmodels.api as sm
    x = (np.asarray(yil, float) - np.mean(yil)) / 10.0
    m = sm.OLS(np.asarray(y, float), sm.add_constant(x)).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags})
    ci = m.conf_int()
    return float(m.params[1]), float(ci[1, 0]), float(ci[1, 1]), float(m.pvalues[1])


def karsilastir(era5, merra2):
    """İki yıllık seriyi ortak dönemde karşılaştırır. Girdi: yıl indeksli Series."""
    ortak = pd.concat([era5.rename("era5_C"), merra2.rename("merra2_C")], axis=1).dropna()
    yil = ortak.index.values
    e, m = ortak["era5_C"].values, ortak["merra2_C"].values

    rows = {}
    for ad, seri in (("ERA5-Land", e), ("MERRA-2", m)):
        s, lo, hi, p = egim_hac(seri, yil)
        rows[ad] = {"egim_C_on_yil": s, "ga_alt": lo, "ga_ust": hi, "p_HAC": p}
    tablo = pd.DataFrame(rows).T

    # Mutlak seviye farkı yeniden analizler arası normaldir; asıl soru eğilimin ve dalgalanmanın uyumu.
    ozet = {
        "ortak_donem": f"{int(yil.min())}-{int(yil.max())}",
        "yil_sayisi": len(yil),
        "ortalama_fark_MERRA2_eksi_ERA5_C": float((m - e).mean()),
        "korelasyon_ham": float(np.corrcoef(e, m)[0, 1]),
        # Eğilimden arındırılmış korelasyon: sadece yıldan yıla dalgalanmanın uyumu
        "korelasyon_trendsiz": float(np.corrcoef(
            e - np.polyval(np.polyfit(yil, e, 1), yil),
            m - np.polyval(np.polyfit(yil, m, 1), yil))[0, 1]),
        "egim_farki_C_on_yil": rows["MERRA-2"]["egim_C_on_yil"] - rows["ERA5-Land"]["egim_C_on_yil"],
    }
    return ortak, tablo, ozet


def analiz():
    import matplotlib.pyplot as plt
    import xarray as xr

    t2m = npz_oku()
    merra2 = yillik_seri(t2m)
    era5_df = pd.read_csv(ERA5_CSV, encoding="utf-8-sig")
    era5 = era5_df.set_index("yil")["ortalama_sicaklik_C"]

    ortak, tablo, ozet = karsilastir(era5, merra2)
    print("\n", tablo.round(4).to_string())
    for k, v in ozet.items():
        print(f"{k}: {v}")

    ortak.reset_index().rename(columns={"index": "yil"}).to_csv(
        DATA_DIR / "merra2_era5_karsilastirma.csv", index=False, encoding="utf-8-sig")
    tablo.reset_index().rename(columns={"index": "veri_seti"}).to_csv(
        DATA_DIR / "merra2_era5_egim_tablosu.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ortak.index, ortak["era5_C"], marker="o", ms=3, label="ERA5-Land (Copernicus)")
    ax.plot(ortak.index, ortak["merra2_C"], marker="s", ms=3, label="MERRA-2 (NASA GMAO)")
    ax.set_title("Türkiye yıllık ortalama 2 m sıcaklık: ERA5-Land ve MERRA-2", fontweight="bold")
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Sıcaklık (°C)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(DATA_DIR / "merra2_era5_karsilastirma.png", dpi=160)
    print("Kaydedildi: data/merra2_era5_*.csv ve .png")


if __name__ == "__main__":
    adim = sys.argv[1] if len(sys.argv) > 1 else "hepsi"
    if adim in ("indir", "hepsi"):
        indir_hizli()
    if adim == "indir_yavas":
        indir_yavas()
    if adim == "indir_yagis":
        indir_hizli(short_name="M2TMNXFLX", degisken="PRECTOTCORR",
                    cikti_ad="merra2_prectotcorr_turkiye.npz")
    if adim in ("analiz", "hepsi"):
        analiz()