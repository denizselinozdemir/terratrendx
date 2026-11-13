from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"

# Yıl aralığı kaydırıcısıyla seçilen dönemlerde istatistik hesaplamak için
# gereken en az yıl sayısı. Daha kısa dönemlerde eğim doğal dalgalanmalardan
# çok etkilenir ve p-değerleri yanıltıcı olur.
MIN_STAT_YEARS = 15

st.set_page_config(page_title="TerraTrendX", page_icon="🌍", layout="wide",
                   initial_sidebar_state="expanded")


# ------------------------- VERİ YÜKLEME -------------------------

@st.cache_data
def load_csv(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


required_files = {
    "turkiye": "turkiye_yillik_sicaklik.csv",
    "bolgeler": "bolgesel_sicaklik_egilimleri.csv",
    "bolgesel_yillik": "bolgesel_yillik_sicaklik.csv",
    "bolgesel_hac": "bolgesel_hac_sonuclari.csv",
    "hac_duyarlilik": "hac_duyarlilik_sonuclari.csv",
    "bolgesel_kirilma": "bolgesel_kirilma_sonuclari.csv",
    "bolge_karsilastirma": "bolgesel_kirilma_sonrasi_karsilastirma.csv",
}
data = {key: load_csv(filename) for key, filename in required_files.items()}
optional_files = {
    "national_break": "kirilma_noktasi_sonuclari.csv",
    "break_bootstrap": "bootstrap_kirilma_sonuclari.csv",
}
optional_data = {key: load_csv(filename) for key, filename in optional_files.items()}

missing = [filename for key, filename in required_files.items() if data[key] is None]
if missing:
    st.error("Aşağıdaki veri dosyaları data klasöründe bulunamadı:")
    for filename in missing:
        st.write(f"- `{filename}`")
    st.info("Dosyaları EarthScope_v2/data klasörüne koyduktan sonra uygulamayı yeniden çalıştırın.")
    st.stop()

turkiye = data["turkiye"]
bolgeler = data["bolgeler"]
bolgesel_yillik = data["bolgesel_yillik"]
bolgesel_hac = data["bolgesel_hac"]
hac_duyarlilik = data["hac_duyarlilik"]
bolgesel_kirilma = data["bolgesel_kirilma"]
bolge_karsilastirma = data["bolge_karsilastirma"]
national_break = optional_data["national_break"]
bootstrap_break = optional_data["break_bootstrap"]


# ------------------------- YARDIMCI FONKSİYONLAR -------------------------

def find_column(frame, candidates):
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def show_table(frame, columns=None):
    if frame is None:
        st.info("Bu veri dosyası mevcut değil.")
        return
    if columns is None:
        selected = frame
    else:
        selected = frame[[column for column in columns if column in frame.columns]]
    st.dataframe(selected, use_container_width=True, hide_index=True)


def hac_slope(series, years, maxlags=3):
    values = pd.to_numeric(series, errors="coerce")
    year_values = pd.to_numeric(years, errors="coerce")
    valid = values.notna() & year_values.notna()
    values = values[valid].to_numpy(dtype=float)
    year_values = year_values[valid].to_numpy(dtype=float)
    if len(values) < 5:
        return None

    x = (year_values - year_values.mean()) / 10.0
    model = sm.OLS(values, sm.add_constant(x)).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": min(maxlags, len(values) - 1)},
    )
    ci = model.conf_int()
    return {
        "slope": float(model.params[1]),
        "lower": float(ci[1, 0]),
        "upper": float(ci[1, 1]),
        "p": float(model.pvalues[1]),
        "r2": float(model.rsquared),
        "n": len(values),
    }


def plot_temperature_trend(frame, year_col, temp_col, title):
    """Yıllık sıcaklık serisini ve doğrusal OLS eğilim çizgisini çizer."""
    plot_data = frame[[year_col, temp_col]].copy()
    plot_data[year_col] = pd.to_numeric(plot_data[year_col], errors="coerce")
    plot_data[temp_col] = pd.to_numeric(plot_data[temp_col], errors="coerce")
    plot_data = plot_data.dropna().sort_values(year_col)
    if len(plot_data) < 2:
        st.warning("Grafik oluşturmak için yeterli veri bulunamadı.")
        return

    years = plot_data[year_col].to_numpy(dtype=float)
    temperatures = plot_data[temp_col].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(years, temperatures, marker="o", markersize=3, linewidth=1.5,
            label="Yıllık ortalama sıcaklık")
    trend = np.polyval(np.polyfit(years, temperatures, 1), years)
    ax.plot(years, trend, linestyle="--", linewidth=2.2,
            label="Doğrusal eğilim (OLS)")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Ortalama hava sıcaklığı (°C)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=9))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_horizontal_bars(frame, label_col, value_col, title, xlabel):
    """Bölge isimlerinin rahat okunması için yatay çubuk grafik çizer."""
    plot_data = frame[[label_col, value_col]].copy()
    plot_data[value_col] = pd.to_numeric(plot_data[value_col], errors="coerce")
    plot_data = plot_data.dropna().sort_values(value_col)
    if plot_data.empty:
        st.warning("Grafik için gösterilebilir sayısal veri bulunamadı.")
        return

    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(plot_data) + 1.5)))
    bars = ax.barh(plot_data[label_col].astype(str), plot_data[value_col])
    ax.axvline(0, linewidth=0.8, alpha=0.6)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    max_abs = max(float(plot_data[value_col].abs().max()), 0.001)
    for bar, value in zip(bars, plot_data[value_col]):
        offset = 0.012 * max_abs
        if value >= 0:
            ax.text(value + offset, bar.get_y() + bar.get_height()/2,
                    f"{value:.3f}", va="center", fontsize=9)
        else:
            ax.text(value - offset, bar.get_y() + bar.get_height()/2,
                    f"{value:.3f}", va="center", ha="right", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(
        ["true", "1", "yes", "evet"]
    )


def explain(title, body):
    """Bir bölümün altına açılır açıklama kutusu ekler."""
    with st.expander(f"ℹ️ {title}"):
        st.markdown(body)


# ------------------------- AÇIKLAMA METİNLERİ -------------------------

QUICK_GUIDE = """
- **Isınma hızı (°C / on yıl):** Sıcaklığın her 10 yılda ortalama ne kadar
  değiştiğini gösterir.
- **Güven aralığı:** Tahminin ne kadar kesin olduğunu gösterir. Aralık ne kadar
  darsa tahmin o kadar kesindir.
- **p-değeri:** Gerçekte eğilim olmasaydı, gözlenen kadar büyük bir eğimi görme
  olasılığıdır. Küçük p-değeri güçlü kanıt demektir ama eğimin büyüklüğünü
  veya nedenini söylemez.
- Her bölümün altında **ℹ️ işaretli açıklama kutuları** vardır; ayrıntı için açabilirsiniz.
"""

EXPL_METRICS = """
**Isınma hızı (eğim).** Yıllık sıcaklıklara uydurulan doğrunun eğimidir:
sıcaklığın 10 yılda ortalama kaç °C değiştiğini gösterir. Örneğin
+0,2 °C/on yıl, 76 yılda yaklaşık 1,5 °C'lik doğrusal bir artış demektir.
Bu değer ilk ve son yılın farkı değildir; tüm yılları birlikte değerlendirir.

**p-değeri.** "Gerçekte eğilim olmasaydı (eğim sıfır olsaydı) bu kadar büyük
bir eğimi görme olasılığı nedir?" sorusunun yanıtıdır. Küçük p-değeri (örneğin
0,05'ten küçük) eğilimin sıfırdan farklı olduğuna dair güçlü kanıt sunar.
p-değeri eğimin büyüklüğünü veya nedenini söylemez.

**R².** Yıllık sıcaklık değişiminin ne kadarının zamanla ilişkili doğrusal
eğilimle açıklandığını gösterir (0 ile 1 arası). R² düşükse bu eğilim olmadığı
anlamına gelmez; yıldan yıla doğal değişkenlik (sıcak ve soğuk yıllar) ve
doğrusal olmayan değişimler de payını alır.

**%95 güven aralığı.** Eğim tahminindeki belirsizliği gösterir. Aralık sıfırı
içermiyorsa eğilim %5 düzeyinde istatistiksel olarak anlamlıdır. Aralık ne
kadar darsa tahmin o kadar kesindir.

**Yıl aralığı kaydırıcısı.** Kısa dönemlerde eğim, doğal yıllık dalgalanmalardan
çok etkilenir. Bu nedenle istatistikler yalnızca en az 15 yıllık seçimlerde
hesaplanır.
"""

EXPL_HAC = """
**Neden HAC?** Ardışık yılların sıcaklıkları birbirine benzeme eğilimindedir
(otokorelasyon). Sıradan regresyon her yılı bağımsız bir gözlem sayar; bu da
belirsizliği olduğundan küçük, sonucu olduğundan kesin gösterir. Newey–West HAC
standart hataları bu bağımlılığı hesaba katar.

**Ne değişir, ne değişmez?** Eğim tahmini aynı kalır; güven aralığı genişler
ve p-değeri büyür.

**Gecikme sayısı.** Kaç yıl öncesine kadar olan bağımlılığın hesaba katılacağını
belirler. Tabloda 1, 3 ve 5 gecikme denenmiştir: gecikme arttıkça aralık
genişler, ancak sonuç her üç seçimde de anlamlı kalmıştır. Bu, sonucun gecikme
seçimine dayanıklı olduğunu gösterir.
"""

EXPL_MODEL = """
Isınma 76 yıl boyunca sabit hızda olmayabilir. Bu nedenle doğrusal, ikinci
dereceden ve parçalı doğrusal modeller karşılaştırılmıştır.

**AIC**, model uyumunu karmaşıklık cezasıyla birlikte ölçer; **düşük AIC daha
iyi uyum** demektir. Doğrusal modelin AIC değeri diğer ikisinden belirgin
biçimde yüksektir: ısınma hızı zaman içinde değişmiştir.

**Dikkat:** İkinci dereceden modelin uç yıllardaki (1950 ve 2025) anlık
eğimleri en az güvenilir tahminlerdir ve bulgu olarak yorumlanmamalıdır.
Ayrıca geçmişte görülen hızlanma, gelecekte de süreceğini tek başına kanıtlamaz.
"""

EXPL_MERRA = """
**Neden ikinci bir veri seti?** Bir yeniden analiz kendi model ve girdi seçimlerini taşır.
Aynı soruyu farklı bir kurumun (NASA GMAO) veri setiyle sormak, sonucun tek bir
veri setinin özelliği olmadığını sınar.

**Eğilimden arındırılmış korelasyon** iki seriden de doğrusal eğilim çıkarıldıktan
sonra hesaplanır; yani yalnızca yıldan yıla sıcak ve soğuk yılların ne kadar örtüştüğünü
ölçer. Yüksek değer, iki veri setinin aynı yılları sıcak/soğuk gördüğünü gösterir.

**Ortalama fark** mutlak sıcaklık düzeyleri arasındaki farktır. Farklı ızgara çözünürlüğü
(MERRA-2 yaklaşık 0,5° × 0,625°, ERA5-Land 0,1°) ve arazi yüksekliği temsili nedeniyle
küçük bir düzey farkı beklenir; eğilim için asıl önemli olan düzey değil değişimdir.

**Neyi göstermez?** İki veri seti birbirinden bağımsız üretilse de ikisi de büyük ölçüde
benzer uydu ve yüzey gözlemlerini kullanır; bu nedenle bu karşılaştırma istasyon
ölçümleriyle doğrulamanın yerini tutmaz. Karşılaştırma yalnızca MERRA-2'nin başladığı
1980'den itibaren yapılabildiği için 1950–1980 dönemini sınamaz.
"""

EXPL_BREAK = """
**Parçalı doğrusal model.** Seri iki ayrı doğru parçasıyla temsil edilir;
doğruların birleştiği yıl "kırılma yılı"dır. Aday yıllar denenir ve AIC'si en
düşük olan seçilir. Her iki tarafta en az 15 yıl bulunması şartı konmuştur.

**Sonuç nasıl okunur?** 1992 öncesinde eğilim hafif negatif, sonrasında belirgin
biçimde pozitiftir. Bu, ısınmanın yaklaşık 1985–1999 arasında hızlandığına
işaret eder; tam yıl belirsizdir.

**Önemli sınırlama.** Kırılma yılı veriye bakılarak seçildiği için, iki eğim
arasındaki farka olağan p-değerleri hesaplamak sonucu olduğundan güçlü gösterir
(seçim sonrası çıkarım sorunu). Bu nedenle uygulama, kırılma farkı için
p-değerini öne çıkarmaz; ana mesaj aşağıdaki bootstrap belirsizlik aralığıdır.
Ayrıca erken dönemde (1950'ler–1980'ler) yeniden analizin dayandığı gözlem
sayısı daha azdır; bu dönemin eğilimleri daha temkinli yorumlanmalıdır.
"""

EXPL_BOOTSTRAP = """
**Bootstrap**, seriyi çok sayıda kez yeniden örnekleyerek (burada 1.000 tekrar)
her seferinde kırılma yılını yeniden bulur. Bulunan yılların dağılımı,
kırılma yılının ne kadar belirsiz olduğunu gösterir.

**Blok yöntemi.** Yıllar tek tek değil, 3, 5 veya 7 yıllık ardışık bloklar
halinde çekilir; böylece ardışık yıllar arasındaki bağımlılık korunur. Üç blok
uzunluğunda da benzer sonuç çıkması, sonucun blok seçimine aşırı duyarlı
olmadığını gösterir.

**%95 aralık.** Kırılma yılının makul olarak hangi yıllar arasında olabileceğini
gösterir. Aralık geniş olduğundan 1992 kesin bir tarih değil, merkezi bir
tahmindir.
"""

EXPL_REGIONAL = """
Her bölge ayrı bir doğrusal eğilim olarak incelenir. Aynı anda yedi test
yapıldığında, en az birinin yalnızca şans eseri "anlamlı" çıkma olasılığı
artar. **Benjamini–Hochberg (BH)** düzeltmesi, bu "çoklu karşılaştırma"
sorununu yanlış keşif oranını kontrol ederek giderir; düzeltilmiş p-değerleri
bu yüzden ham p-değerlerinden büyük veya eşittir.

Bölge eğimlerinin farklı olması, farkların istatistiksel olarak anlamlı olduğu
anlamına gelmez; farklar için aşağıdaki "Bölgeler Arası Isınma Eğilimi
Farkları" bölümüne bakın.
"""

EXPL_REGIONAL_BREAK = """
Her bölge için ulusal analizdeki gibi parçalı doğrusal model kurulmuş ve
kırılma yılı AIC ile seçilmiştir. Çoğu bölgede kırılma yılının ulusal sonuca
yakın çıkması, ısınma hızlanmasının bölgeler arasında ortak bir zaman
özelliği taşıdığına işaret eder. Ancak bu yıllar kesin tarih değil, aday
yıllar arasından seçilmiş tahminlerdir ve ulusal analizdeki belirsizlik
(bootstrap) burada da geçerlidir.
"""

EXPL_PAIRWISE = """
Yedi bölgeden 21 ikili karşılaştırma yapılır. Bu kadar çok test olduğunda
düzeltme şarttır.

- **BH:** Yanlış keşif oranını kontrol eder.
- **BY (Benjamini–Yekutieli):** BH'nin, testlerin birbirine bağlı olabildiği
  durumlar için daha temkinli sürümüdür. Komşu bölgelerin sıcaklıkları birbirine
  bağlı olduğundan BY burada daha uygun kabul edilir; bu yüzden önce BY
  sonuçları gösterilir.

**Anlamlı fark bulunmaması, eğimlerin eşit olduğu anlamına gelmez**; yalnızca
verinin bu farkı güvenle ayırt edemediğini gösterir. Karşılaştırma dönemi
(1992 sonrası) veriye bakılarak seçilmiş bir kırılma yılına dayandığı için
sonuçlar betimleyici kabul edilmelidir.
"""


# ------------------------- YAĞIŞ SAYFASI -------------------------

EXPL_YAGIS = """
**Yıllık toplam yağış (mm).** ERA5-Land'in aylık ortalama günlük yağışı, ayın gün sayısıyla
çarpılıp 12 ay toplanarak yıllık toplama çevrilir. 1 mm yağış, metrekareye 1 litre su demektir.

**Eğilim ve yüzde değişim.** Eğim, yıllık toplamın 10 yılda ortalama kaç mm değiştiğini gösterir
(+ artış, − azalış). Yüzde değişim, bu eğimin dönem ortalamasına oranıdır; ortalama yağışı çok
farklı olan bölgeleri karşılaştırmak için daha uygundur.

**Neden üç yöntem?** Yağış sıcaklıktan farklı olarak çarpık dağılır: bir iki aşırı yağışlı ya da
kurak yıl eğimi belirgin biçimde etkileyebilir. HAC regresyonu (sıcaklıktaki gibi) otokorelasyonu
hesaba katar. Theil–Sen eğimi ve Kendall testi ise uç yıllara karşı daha dayanıklıdır. Üç
yöntem aynı yönü gösteriyorsa sonuç daha güvenilirdir; göstermiyorsa eğilim belirsizdir.

**Yağış neden daha belirsiz?** Yıldan yıla değişkenlik, ortalamaya göre sıcaklıktan çok daha
büyüktür; bu yüzden aynı uzunluktaki seride bir eğilimi yağışta ayırt etmek zordur ve güven
aralıkları geniş çıkar. Eğimin anlamlı olmaması "eğilim yok" demek değildir; yalnızca verinin
bunu güvenle ayırt edemediğini gösterir.
"""


def plot_yagis_serisi(frame, year_col, val_col, title):
    plot_data = frame[[year_col, val_col]].apply(pd.to_numeric, errors="coerce").dropna()
    plot_data = plot_data.sort_values(year_col)
    if len(plot_data) < 2:
        st.warning("Grafik oluşturmak için yeterli veri bulunamadı.")
        return
    x = plot_data[year_col].to_numpy(dtype=float)
    y = plot_data[val_col].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, y, width=0.8, alpha=0.55, label="Yıllık toplam yağış")
    ax.plot(x, np.polyval(np.polyfit(x, y, 1), x), linestyle="--", linewidth=2.2,
            label="Doğrusal eğilim (OLS)")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Toplam yağış (mm / yıl)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=9))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def yagis_sayfasi():
    ulusal = load_csv("yagis_turkiye_yillik.csv")
    ulusal_egilim = load_csv("yagis_ulusal_egilim.csv")
    bolgesel = load_csv("yagis_bolgesel_yillik.csv")
    bolgesel_egilim = load_csv("yagis_bolgesel_egilim.csv")
    if any(f is None for f in (ulusal, ulusal_egilim, bolgesel, bolgesel_egilim)):
        st.info("Yağış analizi henüz hazırlanmadı: `yagis_*.csv` dosyaları data klasöründe bulunamadı.")
        return

    st.header("Türkiye Genelinde Yağış Değişimi")
    st.info("Veri türü: ERA5-Land yeniden analiz verisi · Değişken: toplam yağış (tp) · "
            "Yıllık toplam (mm) · Dönem: 1950–2025")
    row = ulusal_egilim.iloc[0]
    p1, p2, p3 = st.columns(3)
    p1.metric("Ortalama yıllık yağış", f"{row['ortalama_mm']:.0f} mm")
    p2.metric("Uzun dönem eğilimi (HAC)", f"{row['hac_egim_mm_on_yil']:+.1f} mm / 10 yıl")
    p3.metric("Ortalamaya göre değişim", f"{row['yuzde_on_yil']:+.1f} % / 10 yıl")

    year_col, val_col = "yil", "yagis_mm"
    ulusal[year_col] = pd.to_numeric(ulusal[year_col], errors="coerce")
    ulusal[val_col] = pd.to_numeric(ulusal[val_col], errors="coerce")
    ulusal = ulusal.dropna(subset=[year_col, val_col]).sort_values(year_col)
    y0, y1 = int(ulusal[year_col].min()), int(ulusal[year_col].max())

    st.subheader("Türkiye Yıllık Toplam Yağışı")
    secilen = st.slider("İncelenecek yıl aralığını seçin", min_value=y0, max_value=y1,
                        value=(y0, y1), key="yagis_year_range")
    sec = ulusal[ulusal[year_col].between(*secilen)].copy()
    plot_yagis_serisi(sec, year_col, val_col, "Türkiye Yıllık Toplam Yağış (ERA5-Land)")
    st.caption("Çubuklar yıllık toplam yağışı, kesikli çizgi doğrusal eğilimi gösterir. "
               "Yağış yıldan yıla çok dalgalandığı için eğilim çizgisi etrafındaki saçılım büyüktür.")
    sonuc = None
    if len(sec) >= MIN_STAT_YEARS:
        sonuc = hac_slope(sec[val_col], sec[year_col])
    if sonuc:
        s1, s2, s3 = st.columns(3)
        s1.metric("Seçilen dönemin eğimi", f"{sonuc['slope']:+.1f} mm / 10 yıl")
        s2.metric("HAC düzeltilmiş p-değeri", f"{sonuc['p']:.3g}")
        s3.metric("R²", f"{sonuc['r2']:.3f}")
        st.caption(f"HAC %95 güven aralığı: {sonuc['lower']:+.1f} ile {sonuc['upper']:+.1f} mm / 10 yıl. "
                   "Newey–West HAC, 3 gecikme.")
    else:
        st.warning(f"İstatistiksel analiz için en az {MIN_STAT_YEARS} yıllık veri seçilmelidir.")

    st.subheader("Üç yöntemle eğilim (1950–2025)")
    tablo = pd.DataFrame({
        "Yöntem": ["HAC regresyonu", "Theil–Sen eğimi"],
        "Eğim (mm / 10 yıl)": [row["hac_egim_mm_on_yil"], row["sen_egim_mm_on_yil"]],
        "%95 GA alt": [row["hac_ga_alt"], row["sen_ga_alt"]],
        "%95 GA üst": [row["hac_ga_ust"], row["sen_ga_ust"]],
    })
    st.dataframe(tablo.round(2), use_container_width=True, hide_index=True)
    anlamli = "istatistiksel olarak anlamlı" if row["hac_p"] < 0.05 else "istatistiksel olarak anlamlı değil"
    st.caption(f"HAC p-değeri {row['hac_p']:.3g}: ulusal eğilim %5 düzeyinde {anlamli}. "
               f"Kendall tau = {row['kendall_tau']:.3f} (p = {row['kendall_p']:.3g}).")
    explain("Yağış analizi nasıl okunur?", EXPL_YAGIS)
    st.divider()

    st.header("Bölgelere Göre Yağış Eğilimleri")
    show_table(bolgesel_egilim, [
        "bolge", "ortalama_mm", "hac_egim_mm_on_yil", "hac_ga_alt", "hac_ga_ust",
        "yuzde_on_yil", "sen_egim_mm_on_yil", "BH_duzeltilmis_p", "anlamli_mi",
    ])
    plot_horizontal_bars(bolgesel_egilim, "bolge", "yuzde_on_yil",
                         "Bölgelere Göre Yağış Eğilimi (ortalamaya göre değişim)",
                         "Değişim (% / 10 yıl)")
    st.caption("Yedi bölgenin p-değerleri Benjamini–Hochberg (BH) yöntemiyle düzeltilmiştir. "
               "Bölgeler arasında eğim farklarının anlamlılığı yağış için ayrıca test edilmemiştir.")
    bolge_listesi = sorted(bolgesel["bolge"].dropna().unique().tolist())
    secili_bolge = st.selectbox("İncelemek istediğiniz bölgeyi seçin", bolge_listesi, key="yagis_bolge")
    plot_yagis_serisi(bolgesel[bolgesel["bolge"] == secili_bolge], "yil", "yagis_mm",
                      f"{secili_bolge} Yıllık Toplam Yağışı")
    st.divider()

    st.header("Sıcaklık ve Yağış Arasındaki İlişki")
    sic = load_csv("turkiye_yillik_sicaklik.csv")
    s_year = find_column(sic, ["yil", "year"]) if sic is not None else None
    s_temp = find_column(sic, ["ortalama_sicaklik_C", "sicaklik_C", "temperature_C"]) if sic is not None else None
    if s_year and s_temp:
        sic2 = sic[[s_year, s_temp]].rename(columns={s_year: "yil", s_temp: "sicaklik_C"})
        sic2 = sic2.apply(pd.to_numeric, errors="coerce")
        birlesik = pd.merge(ulusal[["yil", "yagis_mm"]], sic2, on="yil").dropna().sort_values("yil")
        if len(birlesik) >= MIN_STAT_YEARS:
            yl = birlesik["yil"].to_numpy(dtype=float)
            tt = birlesik["sicaklik_C"].to_numpy(dtype=float)
            pp = birlesik["yagis_mm"].to_numpy(dtype=float)
            r_ham = float(np.corrcoef(tt, pp)[0, 1])
            tt_d = tt - np.polyval(np.polyfit(yl, tt, 1), yl)
            pp_d = pp - np.polyval(np.polyfit(yl, pp, 1), yl)
            r_dt = float(np.corrcoef(tt_d, pp_d)[0, 1])
            r1, r2 = st.columns(2)
            r1.metric("Korelasyon (ham seriler)", f"{r_ham:+.3f}")
            r2.metric("Korelasyon (eğilimden arındırılmış)", f"{r_dt:+.3f}")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(tt, pp, s=22, alpha=0.8)
            ax.plot(np.sort(tt), np.polyval(np.polyfit(tt, pp, 1), np.sort(tt)), linestyle="--", linewidth=2)
            ax.set_title("Yıllık Ortalama Sıcaklık ve Toplam Yağış", fontsize=14, fontweight="bold", pad=14)
            ax.set_xlabel("Ortalama sıcaklık (°C)")
            ax.set_ylabel("Toplam yağış (mm / yıl)")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.caption(
                "Her nokta bir yıldır. Eğilimden arındırılmış korelasyon, iki seriden de uzun dönem eğilim "
                "çıkarıldıktan sonra hesaplanır; yani sıcak yılların kurak ya da yağışlı olma eğilimini ölçer. "
                "Bu ilişkinin istatistiksel anlamlılığı ayrıca test edilmemiştir ve korelasyon nedensellik "
                "göstermez."
            )
        else:
            st.warning("Sıcaklık ve yağış serileri için yeterli ortak yıl bulunamadı.")
    else:
        st.warning("Sıcaklık serisi okunamadı.")

    with st.expander("Yağış analizinin sınırları"):
        st.markdown(
            """
            - ERA5-Land yağışı bir **model ürünüdür**: atmosferik yeniden analizden gelen yağışla
              çalıştırılır ve istasyon ya da uydu yağış ürünleriyle doğrulanmamıştır. Sıcaklıktan daha
              belirsizdir; dağlık bölgelerde ve kıyılarda sapma gösterebilir.
            - Yıllık yağış, yıldan yıla çok değişkendir. 76 yıllık bir seride bile, onlarca yıllık doğal
              salınımlar eğilim tahminini etkileyebilir.
            - Yalnızca **yıllık toplam** incelenmiştir. Mevsimsel kaymalar, yağış şiddeti, kuraklık
              indeksleri ve aşırı olaylar bu analizin kapsamı dışındadır.
            - Bölgesel eğilimlerin birbirinden farkı yağış için ayrıca test edilmemiştir.
            - Kırılma noktası analizi yalnızca sıcaklık için yapılmıştır.
            - İstatistiksel anlamlılık tek başına neden-sonuç ilişkisini göstermez.
            """
        )
    st.caption(
        "Veri kaynağı: Copernicus Climate Change Service (C3S), ERA5-Land Monthly Averaged Data. "
        "Analiz değişkeni toplam yağıştır (tp). Yeniden analiz verileri doğrudan istasyon "
        "gözlemleriyle aynı şey değildir."
    )


# ------------------------- BAŞLIK VE TANITIM -------------------------

st.title("🌍 TerraTrendX")
st.caption("CLIMATE DATA · TREND ANALYSIS · DISCOVERY")
st.markdown(
    """
    **Türkiye'nin uzun dönemli sıcaklık ve yağış değişimini keşfedin.**

    TerraTrendX, Copernicus ERA5-Land yeniden analiz verilerini kullanarak
    Türkiye'nin ve yedi coğrafi bölgesinin 1950–2025 dönemindeki 2 metre
    hava sıcaklığı ve toplam yağış eğilimlerini inceler.
    """
)
st.divider()
secim = st.radio(
    "Neyi incelemek istersiniz?",
    ["🌡️ Sıcaklık", "🌧️ Yağış"],
    index=None, horizontal=True, key="ana_secim",
)
if secim is None:
    ca, cb = st.columns(2)
    with ca:
        st.markdown(
            "**🌡️ Sıcaklık**\n\n1950–2025 yıllık ortalama hava sıcaklığı: eğilim, "
            "istatistiksel güvenilirlik, kırılma noktası, bölgesel karşılaştırmalar ve "
            "NASA MERRA-2 ile doğrulama."
        )
    with cb:
        st.markdown(
            "**🌧️ Yağış**\n\n1950–2025 yıllık toplam yağış: eğilim, üç yöntemle "
            "karşılaştırma, bölgesel eğilimler ve sıcaklıkla ilişki."
        )
    st.info("Başlamak için yukarıdan bir konu seçin.")
    st.stop()
if secim == "🌧️ Yağış":
    yagis_sayfasi()
    st.stop()

st.info("Veri türü: ERA5-Land yeniden analiz verisi · Değişken: 2 metre hava "
        "sıcaklığı (t2m) · Dönem: 1950–2025")

with st.expander("Bu sayfayı nasıl okumalıyım? (kısa rehber)", expanded=False):
    st.markdown(QUICK_GUIDE)

with st.expander("Araştırma sorusu, veri ve yöntem", expanded=False):
    st.markdown(
        """
        **Araştırma sorusu**

        Türkiye'de 1950–2025 döneminde yıllık ortalama 2 metre hava sıcaklığı
        nasıl değişmiştir? Isınma eğiliminin büyüklüğü ve istatistiksel
        anlamlılığı nedir; bölgeler arasında farklılık var mıdır?

        **Veri**
        - Kaynak: Copernicus Climate Change Service (C3S), ERA5-Land.
        - Değişken: 2 metre hava sıcaklığı (`t2m`).
        - Aylık ortalamalar Kelvin'den Celsius'a çevrilir.
        - Yıllık ortalama, ayların gün sayıları dikkate alınarak hesaplanır.

        **Coğrafi kapsam ve ağırlıklandırma**
        - Türkiye kapsamı yedi coğrafi bölgenin poligonlarının birleşimiyle belirlenir.
        - Izgara hücresi merkezi poligon içinde kalıyorsa hücre analize alınır.
        - Ulusal ve bölgesel ortalamalarda enlem için kosinüs ağırlıkları kullanılır.

        **İstatistiksel yöntemler**
        - Doğrusal eğilimler OLS regresyonuyla tahmin edilir.
        - Standart hatalar Newey–West HAC yöntemiyle düzeltilir.
        - Bölgesel testlerde Benjamini–Hochberg (BH) düzeltmesi uygulanır.
        - İkili bölgesel karşılaştırmalarda BH ve Benjamini–Yekutieli (BY) raporlanır.
        - Kırılma yılı aday parçalı doğrusal modeller arasında AIC ile seçilir.
        - Kırılma belirsizliği 3, 5 ve 7 yıllık bloklarla bootstrap kullanılarak incelenir.
        - Sonuçlar, NASA MERRA-2 yeniden analizinden aynı yöntemle çıkarılan ikinci bir
          seriyle 1980–2025 döneminde karşılaştırılır.
        """
    )
st.divider()


# ------------------------- TÜRKİYE GENELİ -------------------------

st.header("Türkiye Genelinde Sıcaklık Değişimi")
national_year_col = find_column(turkiye, ["yil", "year"])
national_temp_col = find_column(
    turkiye, ["ortalama_sicaklik_C", "sicaklik_C", "temperature_C"]
)
if national_year_col is None or national_temp_col is None:
    st.error("Türkiye yıllık sıcaklık CSV sütunları beklenen formatta değil.")
    st.write("Bulunan sütunlar:", list(turkiye.columns))
    st.stop()

turkiye[national_year_col] = pd.to_numeric(turkiye[national_year_col], errors="coerce")
turkiye[national_temp_col] = pd.to_numeric(turkiye[national_temp_col], errors="coerce")
turkiye = turkiye.dropna(subset=[national_year_col, national_temp_col]).sort_values(
    national_year_col
)
start_year = int(turkiye[national_year_col].min())
end_year = int(turkiye[national_year_col].max())

national_full = hac_slope(turkiye[national_temp_col], turkiye[national_year_col])

m1, m2, m3 = st.columns(3)
m1.metric("Başlangıç yılı", str(start_year))
m2.metric("Son yıl", str(end_year))
m3.metric(
    "Uzun dönem eğilimi",
    f"{national_full['slope']:+.3f} °C / on yıl" if national_full else "—",
)

st.subheader("Türkiye Yıllık Ortalama Sıcaklığı")
selected_range = st.slider(
    "İncelenecek yıl aralığını seçin",
    min_value=start_year, max_value=end_year,
    value=(start_year, end_year), key="national_year_range",
)
filtered_national = turkiye[
    turkiye[national_year_col].between(*selected_range)
].copy()
plot_temperature_trend(
    filtered_national, national_year_col, national_temp_col,
    "Türkiye Yıllık Ortalama 2 Metre Hava Sıcaklığı",
)
st.caption(
    "Düz çizgi yıllık ortalama sıcaklığı, kesikli çizgi doğrusal eğilimi "
    "gösterir. Çizginin etrafındaki dalgalanma doğal yıllık değişkenliktir."
)

national_result = None
if len(filtered_national) >= MIN_STAT_YEARS:
    national_result = hac_slope(
        filtered_national[national_temp_col], filtered_national[national_year_col]
    )
if national_result:
    c1, c2, c3 = st.columns(3)
    c1.metric("Seçilen dönemin ısınma hızı",
              f"{national_result['slope']:+.3f} °C / 10 yıl")
    c2.metric("HAC düzeltilmiş p-değeri", f"{national_result['p']:.3e}")
    c3.metric("R²", f"{national_result['r2']:.3f}")
    st.caption(
        "HAC %95 güven aralığı: "
        f"{national_result['lower']:+.3f} ile {national_result['upper']:+.3f} "
        "°C / 10 yıl. Newey–West HAC, 3 gecikme."
    )
else:
    st.warning(
        f"İstatistiksel analiz için en az {MIN_STAT_YEARS} yıllık veri "
        "seçilmelidir. Kısa dönemlerde eğim doğal dalgalanmalardan çok etkilenir."
    )
explain("Bu sayıları nasıl yorumlamalıyım?", EXPL_METRICS)
st.divider()


# ------------------------- HAC DUYARLILIK -------------------------

st.header("İstatistiksel Güvenilirlik Kontrolü")
st.write("Türkiye genelindeki ısınma eğiliminin farklı Newey–West HAC gecikmelerinde nasıl değiştiği.")
show_table(hac_duyarlilik)

lag_col = find_column(hac_duyarlilik, ["HAC gecikmesi", "hac_gecikmesi"])
slope_col = find_column(
    hac_duyarlilik,
    ["Isınma hızı (°C/on yıl)", "isinma_hizi_C_on_yil", "egim_C_on_yil"],
)
if lag_col and slope_col:
    sensitivity = hac_duyarlilik[[lag_col, slope_col]].copy()
    sensitivity[lag_col] = pd.to_numeric(sensitivity[lag_col], errors="coerce")
    sensitivity[slope_col] = pd.to_numeric(sensitivity[slope_col], errors="coerce")
    sensitivity = sensitivity.dropna().sort_values(lag_col)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sensitivity[lag_col], sensitivity[slope_col], marker="o", linewidth=2)
    ax.set_title("HAC Gecikmesine Göre Isınma Hızı", fontweight="bold")
    ax.set_xlabel("HAC gecikme sayısı")
    ax.set_ylabel("Isınma hızı (°C / on yıl)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
st.caption(
    "Farklı HAC gecikme değerlerinde eğim tahmini yaklaşık +0,209 °C/on yıl "
    "düzeyinde sabit kalırken güven aralıkları ve p-değerleri değişebilmektedir."
)
explain("HAC düzeltmesi nedir, neden gerekli?", EXPL_HAC)
st.divider()


# ------------------------- NASA MERRA-2 DOĞRULAMASI -------------------------

merra_series = load_csv("merra2_era5_karsilastirma.csv")
merra_slopes = load_csv("merra2_era5_egim_tablosu.csv")
merra_ok = (
    merra_series is not None and merra_slopes is not None
    and {"yil", "era5_C", "merra2_C"}.issubset(merra_series.columns)
    and {"veri_seti", "egim_C_on_yil", "ga_alt", "ga_ust"}.issubset(merra_slopes.columns)
)
if merra_ok:
    st.header("Başka Bir Veri Seti ile Doğrulama: NASA MERRA-2")
    ms = merra_series.dropna().sort_values("yil")
    yrs = ms["yil"].to_numpy(dtype=float)
    e_vals = ms["era5_C"].to_numpy(dtype=float)
    m_vals = ms["merra2_C"].to_numpy(dtype=float)
    period = f"{int(yrs.min())}–{int(yrs.max())}"
    st.write(
        "Sonuçlarımız tek bir veri setine bağlı kalmasın diye aynı bölge ve aynı yıllık "
        "yöntemle, NASA'nın MERRA-2 yeniden analizinden (2 metre hava sıcaklığı, "
        f"M2TMNXSLV) ikinci bir seri çıkardık. Ortak dönem: {period}."
    )

    def _detrend(v):
        return v - np.polyval(np.polyfit(yrs, v, 1), yrs)

    corr_raw = float(np.corrcoef(e_vals, m_vals)[0, 1])
    corr_dt = float(np.corrcoef(_detrend(e_vals), _detrend(m_vals))[0, 1])
    offset = float((m_vals - e_vals).mean())
    v1, v2, v3 = st.columns(3)
    v1.metric("Korelasyon (ham seri)", f"{corr_raw:.3f}")
    v2.metric("Korelasyon (eğilimden arındırılmış)", f"{corr_dt:.3f}")
    v3.metric("Ortalama fark (MERRA-2 − ERA5-Land)", f"{offset:+.2f} °C")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(yrs, e_vals, marker="o", markersize=3, linewidth=1.5, label="ERA5-Land (Copernicus)")
    ax.plot(yrs, m_vals, marker="s", markersize=3, linewidth=1.5, label="MERRA-2 (NASA)")
    ax.set_title(f"Türkiye Yıllık Ortalama 2 Metre Sıcaklığı: İki Veri Seti ({period})",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Yıl")
    ax.set_ylabel("Ortalama hava sıcaklığı (°C)")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    slope_view = merra_slopes.rename(columns={
        "veri_seti": "Veri seti",
        "egim_C_on_yil": "Isınma hızı (°C/on yıl)",
        "ga_alt": "%95 GA alt",
        "ga_ust": "%95 GA üst",
        "p_HAC": "HAC p-değeri",
    })
    st.dataframe(slope_view, use_container_width=True, hide_index=True)

    lows = merra_slopes["ga_alt"].astype(float)
    highs = merra_slopes["ga_ust"].astype(float)
    if lows.max() <= highs.min():
        st.caption(
            f"{period} döneminde iki veri setinin eğim tahminlerinin güven aralıkları "
            "örtüşüyor: eğimler istatistiksel olarak birbirinden ayırt edilemiyor. "
            "Bu resmî bir fark testi değil, yalnızca aralıkların karşılaştırmasıdır."
        )
    else:
        st.caption(
            f"{period} döneminde iki veri setinin eğim tahminlerinin güven aralıkları "
            "örtüşmüyor: büyüklük konusunda veri setleri arasında fark var. "
            "Eğilimin yönü ve diğer sonuçlar için bu farkın etkisi ayrıca değerlendirilmelidir."
        )
    st.caption(
        f"Eğimler yalnızca {period} dönemine aittir; 1950–2025 uzun dönem eğiliminden "
        "(yaklaşık +0,209 °C/on yıl) daha yüksektir çünkü 1990'lardan sonraki hızlanmayı "
        "içerir ve 1950–1992 arasındaki hafif soğumayı içermez."
    )
    explain("Bu doğrulama neyi gösterir, neyi göstermez?", EXPL_MERRA)
    st.divider()


# ------------------------- BÖLGESEL HAC -------------------------

st.header("Bölgesel Isınma Eğilimlerinin İstatistiksel Analizi")
st.write(
    "Her coğrafi bölgenin 1950–2025 dönemindeki doğrusal sıcaklık eğilimi "
    "Newey–West HAC yöntemiyle incelenmiştir. Yedi bölgeye ait p-değerleri "
    "Benjamini–Hochberg (BH) yöntemiyle düzeltilmiştir."
)
show_table(
    bolgesel_hac,
    ["bolge", "isinma_hizi_C_on_yil", "guven_araligi_alt",
     "guven_araligi_ust", "BH_duzeltilmis_p", "anlamli_mi"],
)
region_hac_slope_col = find_column(
    bolgesel_hac, ["isinma_hizi_C_on_yil", "egilim_C_on_yil"]
)
region_hac_name_col = find_column(bolgesel_hac, ["bolge"])
if region_hac_slope_col and region_hac_name_col:
    plot_horizontal_bars(
        bolgesel_hac, region_hac_name_col, region_hac_slope_col,
        "Bölgelere Göre HAC Düzeltilmiş Isınma Hızları",
        "Isınma hızı (°C / on yıl)",
    )
st.caption(
    "Önceki analizde yedi bölgenin tamamında pozitif ve BH düzeltmesinden "
    "sonra istatistiksel olarak anlamlı eğilim bulunmuştur."
)
explain("Bölgesel sonuçlar ve çoklu karşılaştırma düzeltmesi", EXPL_REGIONAL)
st.divider()


# ------------------------- ULUSAL KIRILMA VE BOOTSTRAP -------------------------

st.header("Türkiye Genelinde Kırılma Noktası Analizi")
st.write(
    "Türkiye'nin yıllık ortalama sıcaklık serisinde iki farklı döneme izin veren "
    "parçalı doğrusal model incelenmiş; aday kırılma yılı AIC ile seçilmiştir."
)

st.subheader("Isınma sabit hızda mı? Model karşılaştırması")
# Not: Aşağıdaki değerler analiz çıktısından alınmış sabit değerlerdir.
# Analiz yeniden çalıştırılırsa burası da güncellenmelidir.
model_comparison = pd.DataFrame({
    "Model": ["Doğrusal", "İkinci dereceden", "Parçalı doğrusal (kırılma: 1992)"],
    "R²": [0.2845, 0.5311, 0.5331],
    "AIC": [171.17, 141.05, 140.72],
})
st.dataframe(model_comparison, use_container_width=True, hide_index=True)
st.caption(
    "Düşük AIC daha iyi uyum demektir. İkinci dereceden ve parçalı doğrusal "
    "modeller doğrusal modelden belirgin biçimde daha iyi uyum sağlamıştır."
)
explain("Model karşılaştırması nasıl okunur?", EXPL_MODEL)

if national_break is not None:
    st.subheader("Ulusal kırılma noktası model sonuçları")
    aic_col = find_column(national_break, ["AIC", "aic"])
    if aic_col:
        national_break[aic_col] = pd.to_numeric(national_break[aic_col], errors="coerce")
        st.caption("Tablo en düşük AIC değerinden başlayarak sıralanmıştır.")
        show_table(national_break.sort_values(aic_col).head(10))
        break_col = find_column(
            national_break,
            ["kirilma_yili", "kırılma_yılı", "break_year", "en_iyi_kirilma_yili"],
        )
        if break_col:
            valid = national_break.dropna(subset=[break_col, aic_col])
            if not valid.empty:
                best = valid.loc[valid[aic_col].idxmin()]
                st.metric("AIC'ye göre seçilen kırılma yılı", f"{int(best[break_col])}")
                st.caption(f"En düşük AIC: {best[aic_col]:.3f}")
    else:
        show_table(national_break)
else:
    st.warning("`kirilma_noktasi_sonuclari.csv` dosyası bulunamadı.")
explain("Kırılma noktası analizi nasıl okunur?", EXPL_BREAK)

st.subheader("Kırılma yılının bootstrap belirsizliği")
if bootstrap_break is not None:
    show_table(bootstrap_break)
    st.caption(
        "Bootstrap aralığı, kullanılan blok uzunluğu ve bootstrap kurulumuna koşulludur."
    )
else:
    st.warning("`bootstrap_kirilma_sonuclari.csv` dosyası bulunamadı.")

bootstrap_plot = DATA_DIR / "bootstrap_kirilma_dagilimi.png"
if bootstrap_plot.exists():
    st.image(str(bootstrap_plot),
             caption="Bootstrap tekrarlarında seçilen kırılma yıllarının dağılımı",
             use_container_width=True)
break_plot = DATA_DIR / "kirilma_noktasi_grafigi.png"
if break_plot.exists():
    st.image(str(break_plot),
             caption="Türkiye yıllık sıcaklık serisi ve parçalı doğrusal model",
             use_container_width=True)
st.caption(
    "Ulusal analizde AIC ile seçilen kırılma yılı 1992'dir. 3, 5 ve 7 yıllık "
    "blok bootstrap sonuçları yaklaşık 1985–1999 aralığında belirsizliğe işaret "
    "eder. Kırılma yılı kesin bir fiziksel geçiş tarihi veya nedensel kanıt değildir."
)
explain("Bootstrap nedir, belirsizliği nasıl gösterir?", EXPL_BOOTSTRAP)
st.divider()


# ------------------------- BÖLGESEL KIRILMA -------------------------

st.header("Bölgesel Sıcaklık Eğilimlerindeki Kırılma Noktaları")
st.write(
    "Bölgelerin yıllık sıcaklık serilerine parçalı doğrusal modeller uygulanmış "
    "ve aday kırılma yılları AIC ölçütüne göre seçilmiştir."
)
regional_break_col = find_column(
    bolgesel_kirilma, ["kirilma_yili", "kırılma_yılı", "break_year"]
)
if regional_break_col:
    display_breaks = bolgesel_kirilma.copy()
    display_breaks[regional_break_col] = pd.to_numeric(
        display_breaks[regional_break_col], errors="coerce"
    )
    show_table(display_breaks.sort_values(regional_break_col))
else:
    st.warning("Kırılma yılı sütunu tanınamadı.")
    show_table(bolgesel_kirilma)
st.caption(
    "Kırılma yılları adaylar arasından seçilmiştir; kesin fiziksel veya nedensel "
    "değişim tarihleri olarak yorumlanmamalıdır."
)
explain("Bölgesel kırılma yılları nasıl yorumlanır?", EXPL_REGIONAL_BREAK)
st.divider()


# ------------------------- BÖLGESEL EĞİM FARKLARI -------------------------

st.header("Bölgeler Arası Isınma Eğilimi Farkları")
st.write(
    "Yedi bölgenin seçilmiş kırılma yıllarından sonraki ortak döneminde "
    "doğrusal ısınma eğimleri ikili olarak karşılaştırılmıştır."
)
comparison_required = [
    "bolge1", "bolge2", "egim_farki_C_on_yil",
    "p_BH", "p_BY", "anlamli_BH", "anlamli_BY",
]
if all(col in bolge_karsilastirma.columns for col in comparison_required):
    if {"ortak_baslangic", "ortak_bitis"}.issubset(bolge_karsilastirma.columns):
        starts = bolge_karsilastirma["ortak_baslangic"].dropna().unique()
        ends = bolge_karsilastirma["ortak_bitis"].dropna().unique()
        if len(starts) == 1 and len(ends) == 1:
            st.caption(
                f"Ortak karşılaştırma dönemi: {int(starts[0])}–{int(ends[0])}. "
                "Tüm 21 bölge çifti aynı dönemde incelenmiştir."
            )

    significant_by = bolge_karsilastirma[
        to_bool(bolge_karsilastirma["anlamli_BY"])
    ].copy()
    st.subheader("BY Düzeltmesine Göre Anlamlı Farklar")
    if not significant_by.empty:
        show_table(significant_by, [
            "bolge1", "bolge2", "egim_farki_C_on_yil",
            "guven_alt", "guven_ust", "p_BY",
        ])
    else:
        st.info("BY düzeltmesinden sonra anlamlı bölgesel eğim farkı bulunmadı.")

    significant_bh = bolge_karsilastirma[
        to_bool(bolge_karsilastirma["anlamli_BH"])
    ].copy()
    with st.expander("BH düzeltmesine göre anlamlı karşılaştırmalar"):
        if not significant_bh.empty:
            show_table(significant_bh, [
                "bolge1", "bolge2", "egim_farki_C_on_yil", "p_BH",
            ])
        else:
            st.info("BH düzeltmesinden sonra anlamlı fark bulunmadı.")

    with st.expander("21 bölge karşılaştırmasının tamamı"):
        show_table(bolge_karsilastirma, [
            "bolge1", "bolge2", "ortak_baslangic", "ortak_bitis",
            "gozlem_sayisi", "egim_farki_C_on_yil", "guven_alt",
            "guven_ust", "p_degeri", "p_BH", "p_BY",
            "anlamli_BH", "anlamli_BY",
        ])
else:
    st.error("Bölgesel karşılaştırma CSV sütunları beklenen formatta değil.")
    st.write(list(bolge_karsilastirma.columns))
st.caption(
    "HAC standart hataları 3 gecikme ile hesaplanmıştır. BH ve BY düzeltmeleri "
    "21 ikili karşılaştırmaya uygulanmıştır. Güven aralıkları tekil karşılaştırmalara "
    "aittir; eşzamanlı çoklu karşılaştırma güven aralıkları değildir."
)
explain("BH ve BY düzeltmeleri neden gerekli?", EXPL_PAIRWISE)
st.divider()


# ------------------------- ETKİLEŞİMLİ BÖLGESEL ANALİZ -------------------------

st.header("Bölgesel Sıcaklık Analizi")
region_year_col = find_column(bolgesel_yillik, ["yil", "year"])
region_temp_col = find_column(
    bolgesel_yillik, ["ortalama_sicaklik_C", "sicaklik_C"]
)
region_name_col_series = find_column(bolgesel_yillik, ["bolge", "BOLGE_ADI"])

if region_name_col_series is None or region_year_col is None or region_temp_col is None:
    st.error("Bölgesel yıllık veri sütunları beklenen formatta değil.")
    st.write("Bulunan sütunlar:", list(bolgesel_yillik.columns))
    st.stop()

region_names = sorted(
    bolgesel_yillik[region_name_col_series].dropna().unique().tolist()
)
selected_region = st.selectbox("İncelemek istediğiniz bölgeyi seçin", region_names)

selected_region_data = bolgesel_yillik[
    bolgesel_yillik[region_name_col_series] == selected_region
].copy()
selected_region_data[region_year_col] = pd.to_numeric(
    selected_region_data[region_year_col], errors="coerce"
)
selected_region_data[region_temp_col] = pd.to_numeric(
    selected_region_data[region_temp_col], errors="coerce"
)
selected_region_data = selected_region_data.dropna(
    subset=[region_year_col, region_temp_col]
).sort_values(region_year_col)

region_min_year = int(selected_region_data[region_year_col].min())
region_max_year = int(selected_region_data[region_year_col].max())
region_range = st.slider(
    "Bölgesel analiz için yıl aralığını seçin",
    min_value=region_min_year, max_value=region_max_year,
    value=(region_min_year, region_max_year), key="regional_year_range",
)
filtered_region = selected_region_data[
    selected_region_data[region_year_col].between(*region_range)
].copy()

st.subheader(f"{selected_region} Yıllık Ortalama Sıcaklığı")
plot_temperature_trend(
    filtered_region, region_year_col, region_temp_col,
    f"{selected_region} Yıllık Ortalama Hava Sıcaklığı",
)

region_result = None
if len(filtered_region) >= MIN_STAT_YEARS:
    region_result = hac_slope(
        filtered_region[region_temp_col], filtered_region[region_year_col]
    )
if region_result:
    r1, r2, r3 = st.columns(3)
    r1.metric("Ortalama sıcaklık",
              f"{filtered_region[region_temp_col].mean():.2f} °C")
    change = (
        filtered_region[region_temp_col].iloc[-1]
        - filtered_region[region_temp_col].iloc[0]
    )
    r2.metric("İlk-son yıl farkı", f"{change:+.2f} °C")
    r3.metric("HAC ısınma hızı", f"{region_result['slope']:+.3f} °C / 10 yıl")
    st.caption(
        f"R²: {region_result['r2']:.3f} · HAC p-değeri: {region_result['p']:.3e} · "
        f"%95 GA: [{region_result['lower']:+.3f}, "
        f"{region_result['upper']:+.3f}] °C / 10 yıl"
    )
    st.caption(
        "İlk-son yıl farkı yalnızca iki yılın karşılaştırmasıdır ve yıllık "
        "dalgalanmadan etkilenir; ısınma hızı ise tüm yılları birlikte kullanır."
    )
else:
    st.warning(
        f"İstatistiksel analiz için en az {MIN_STAT_YEARS} yıllık veri "
        "seçilmelidir. Kısa dönemlerde eğim doğal dalgalanmalardan çok etkilenir."
    )
explain("Bu bölüm nasıl yorumlanır?",
        "Bu bölüm, seçtiğiniz bölge ve dönem için sıcaklık serisini ve "
        "HAC düzeltmeli doğrusal eğilimi gösterir. Sayıların anlamı için "
        "yukarıdaki **\"Bu sayıları nasıl yorumlamalıyım?\"** kutusuna bakın.")
st.divider()


# ------------------------- BÖLGESEL ISINMA HIZLARI -------------------------

st.subheader("Bölgelerin 1950–2025 Isınma Hızları")
summary_region_name = find_column(bolgeler, ["bolge", "BOLGE_ADI"])
summary_slope = find_column(
    bolgeler, ["egilim_C_on_yil", "isinma_hizi_C_on_yil"]
)
if summary_region_name and summary_slope:
    plot_horizontal_bars(
        bolgeler, summary_region_name, summary_slope,
        "Bölgelerin Uzun Dönemli Isınma Hızları",
        "Isınma hızı (°C / on yıl)",
    )
else:
    st.warning("Bölgesel özet tablosunda eğim sütunu tanınamadı.")
    show_table(bolgeler)


# ------------------------- HARİTA -------------------------

st.subheader("Türkiye Bölgesel Sıcaklık Haritası")
map_path = DATA_DIR / "turkiye_bolgesel_sicaklik_haritasi.png"
if map_path.exists():
    st.image(str(map_path), caption="1950–2025 bölgesel sıcaklık eğilimleri",
             use_container_width=True)
else:
    st.warning("Bölgesel sıcaklık haritası PNG dosyası data klasöründe bulunamadı.")
st.caption(
    "Harita, her bölgenin 1950–2025 doğrusal ısınma hızını gösterir. Bölgeler "
    "arasındaki farkların çoğu istatistiksel olarak ayırt edilemez; anlamlı "
    "farklar için \"Bölgeler Arası Isınma Eğilimi Farkları\" bölümüne bakın."
)


# ------------------------- BULGULAR VE SINIRLILIKLAR -------------------------

st.header("Genel Bulgular ve Bilimsel Yorum")
st.markdown(
    """
    ERA5-Land yeniden analiz verisine dayalı ulusal seride, 1950–2025 dönemi
    için doğrusal eğilim yaklaşık **+0,209 °C/on yıl** olarak tahmin edilmiştir.
    Newey–West HAC düzeltmeli %95 güven aralığı **+0,107 ile +0,311 °C/on yıl**;
    p-değeri ise yaklaşık **0,000062**'dir.

    AIC ölçütüyle seçilen ulusal parçalı doğrusal modelin kırılma yılı **1992**'dir.
    Modelde kırılma öncesi eğim yaklaşık **−0,119 °C/on yıl**, kırılma sonrası
    eğim yaklaşık **+0,678 °C/on yıl** bulunmuştur. Blok bootstrap sonuçları
    kırılma yılının belirsizliğe sahip olduğunu, yaklaşık **1985–1999** aralığının
    makul olduğunu göstermektedir.

    Bölgesel analizlerde yedi bölgenin tamamında pozitif uzun dönemli doğrusal
    eğilim bulunmuştur. Bununla birlikte, bölgelerin eğimlerinin birbirinden
    farklı olması her ikili farkın istatistiksel olarak anlamlı olduğu anlamına gelmez.
    """
)
with st.expander("Yorumlama sınırları ve dikkat edilmesi gerekenler"):
    st.markdown(
        """
        - ERA5-Land bir **yeniden analiz veri setidir**; doğrudan meteoroloji
          istasyonu ölçümlerinin eş anlamlısı değildir.
        - Erken dönemlerde (1950'ler–1980'ler) yeniden analizin dayandığı gözlem
          sayısı daha azdır; bu dönemin eğilimleri daha temkinli yorumlanmalıdır.
        - İncelenen değişken yer yüzeyi sıcaklığı değil, yerden yaklaşık
          2 metre yükseklikteki hava sıcaklığıdır.
        - Türkiye maskesi ızgara hücresi merkezlerinin bölge poligonlarında
          bulunmasına dayanır. Kıyı çizgisi ve küçük geometrilerde bu yaklaşım
          mekânsal temsil hatası oluşturabilir.
        - Kosinüs enlem ağırlığı yaklaşık hücre alanı ağırlığı sağlar;
          istasyon yoğunluğu veya yerel ölçüm kalitesi için düzeltme değildir.
        - HAC düzeltmesi zaman serisindeki otokorelasyon ve değişen varyansa
          karşı standart hataları düzenler; mekânsal bağımlılığı tek başına çözmez.
          Yedi bölge birbirine komşu olduğundan bölgesel testler bağımsız değildir.
        - AIC ile seçilen kırılma yılı aday yıllar ve kurulan model ailesi içinde
          en iyi uyumu verir; tek başına fiziksel geçişin veya belirli bir olayın
          nedenini kanıtlamaz.
        - Kırılma yılı veriye bakılarak seçildiğinden, kırılma sonrası eğimlere ve
          eğim farklarına ait p-değerleri seçim sonrası çıkarım nedeniyle olduğundan
          güçlü görünebilir; bu sonuçlar betimleyici kabul edilmelidir.
        - Bootstrap aralığı kullanılan blok uzunluklarına ve kurulan modele
          koşulludur; kırılma yılı seçiminin tüm belirsizliklerini kapsamaz.
        - İkinci dereceden modelin uç yıllardaki eğimleri güvenilir değildir ve
          geçmişteki hızlanma, gelecekte de süreceğini kanıtlamaz.
        - İstatistiksel anlamlılık tek başına neden-sonuç ilişkisini göstermez.
        """
    )
st.divider()


# ------------------------- VERİ İNDİRME -------------------------

st.header("Verileri İndir")
st.write(
    "Uygulamada kullanılan yıllık sıcaklık serilerini ve istatistiksel "
    "sonuç tablolarını CSV biçiminde indirebilirsiniz. Dosyalar Excel'de "
    "Türkçe karakterlerin doğru açılması için UTF-8-SIG kodlamasıyla hazırlanır."
)

def csv_download_button(label, frame, filename, key):
    if frame is None:
        st.info(f"{label}: veri dosyası bulunamadı.")
        return
    csv_bytes = frame.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        key=key,
        use_container_width=True,
    )

download_col1, download_col2 = st.columns(2)
with download_col1:
    csv_download_button(
        "Türkiye yıllık sıcaklık verisini indir",
        turkiye,
        "turkiye_yillik_sicaklik.csv",
        "download_national",
    )
    csv_download_button(
        "Bölgesel yıllık sıcaklık verilerini indir",
        bolgesel_yillik,
        "bolgesel_yillik_sicaklik.csv",
        "download_regional_annual",
    )
    csv_download_button(
        "Bölgesel HAC sonuçlarını indir",
        bolgesel_hac,
        "bolgesel_hac_sonuclari.csv",
        "download_regional_hac",
    )
with download_col2:
    csv_download_button(
        "Bölgesel eğilim özetini indir",
        bolgeler,
        "bolgesel_sicaklik_egilimleri.csv",
        "download_regional_summary",
    )
    csv_download_button(
        "HAC duyarlılık sonuçlarını indir",
        hac_duyarlilik,
        "hac_duyarlilik_sonuclari.csv",
        "download_hac_sensitivity",
    )
    csv_download_button(
        "Bölgesel eğim karşılaştırmalarını indir",
        bolge_karsilastirma,
        "bolgesel_kirilma_sonrasi_karsilastirma.csv",
        "download_pairwise",
    )

optional_col1, optional_col2 = st.columns(2)
with optional_col1:
    if national_break is not None:
        csv_download_button(
            "Ulusal kırılma analizi sonuçlarını indir",
            national_break,
            "kirilma_noktasi_sonuclari.csv",
            "download_national_break",
        )
with optional_col2:
    if bootstrap_break is not None:
        csv_download_button(
            "Bootstrap sonuçlarını indir",
            bootstrap_break,
            "bootstrap_kirilma_sonuclari.csv",
            "download_bootstrap",
        )

st.caption(
    "İndirilen CSV dosyaları analiz tablolarını içerir; ham ERA5-Land NetCDF "
    "verisinin yerine geçmez. Grafikler ve istatistiksel sonuçlar uygulamada "
    "gösterilen analiz kapsamına göre yorumlanmalıdır."
)
st.divider()


# ------------------------- HAM TABLOLAR VE KAYNAK -------------------------

with st.expander("Analiz verilerini görüntüle"):
    st.subheader("Türkiye yıllık sıcaklık verileri")
    show_table(turkiye)
    st.subheader("Bölgesel yıllık sıcaklık verileri")
    show_table(bolgesel_yillik)
    st.subheader("Bölgesel eğilim özeti")
    show_table(bolgeler)

st.divider()
st.caption(
    "Veri kaynağı: Copernicus Climate Change Service (C3S), ERA5-Land Monthly "
    "Averaged Data. Analiz değişkeni 2 metre hava sıcaklığıdır (t2m). Yeniden "
    "analiz verileri doğrudan istasyon gözlemleriyle aynı şey değildir."
)