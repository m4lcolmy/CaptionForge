# CaptionForge — Uygulama Nasıl Çalışır

Uygulamanın iç işleyişine dair çalışan bir rehber: hangi katman ne yapıyor, her
komutta ne oluyor ve çıktıyı hangi kurallar belirliyor. Sürüm 0.7.0, Python
3.12+.

---

## 1. Uygulama gerçekte ne yapıyor

CaptionForge tek bir YouTube video URL'sini altyazı/transkript dosyalarına
(`srt`, `vtt`, `txt`, `json`, `docx`) dönüştürür. İki metin kaynağı vardır ve her zaman
ucuz olanı tercih eder:

1. **Mevcut YouTube altyazıları** — yalnızca altyazı izi indirilir, medya
   indirilmez.
2. **Yerel transkripsiyon** — yalnızca istenen dilde eşleşen bir altyazı yoksa
   (veya `--force` verildiyse). Sadece ses akışı indirilir, mono 16 kHz PCM WAV'a
   dönüştürülür ve yerel makinede `faster-whisper`'a verilir.

Hiçbir yolda video akışı indirilmez. YouTube (`yt-dlp` üzerinden) ve ilk model
indirmesinde Whisper model sunucusu dışında hiçbir servise ağ çağrısı yapılmaz.

---

## 2. Katman haritası

```
app/
├── main.py              giriş noktası → app.interfaces.cli:app  ("captionforge" konsol betiği)
├── interfaces/cli.py    Typer komutları, Rich çıktısı, hata→çıkış kodu eşlemesi
├── services/            orkestrasyon; iş akışının kararlaştırıldığı tek yer
│   ├── video_service.py         URL doğrulama + canlı/erişilebilirlik kontrolleri + keşif
│   ├── subtitle_service.py      iz seçimi, altyazı ayrıştırma, asgari temizlik
│   ├── audio_service.py         iş alanı, ses indirme, FFmpeg dönüştürme
│   ├── transcription_service.py altyazı-öncelikli akış, Whisper yedeği, dışa aktarım
│   ├── postprocessing_service.py  zamanlama + metin normalizasyonu (tüm kaynaklar için ortak)
│   └── export_service.py        format doğrulama, dosya adı, atomik çok formatlı yazma
├── adapters/            dış dünyayla konuşan her şey
│   ├── ytdlp_adapter.py    meta veri, altyazı indirme, ses indirme, hata çevirisi
│   ├── ffmpeg_adapter.py   alt süreç (shell yok), dönüştürme, hata çevirisi
│   └── whisper_adapter.py  tembel faster-whisper importu, cihaz/hesaplama seçimi
├── models/              donmuş (frozen) Pydantic sözleşmeleri (VideoMetadata, SubtitleTrack, …)
├── exporters/           saf render fonksiyonları: segmentler → metin
├── utils/               saf yardımcılar (URL, zaman, dil, dosya adı, Arapça metin)
└── core/                yapılandırma, sabitler, istisna hiyerarşisi, retry, loglama
```

Bağımlılık yönü katıdır: `cli → services → adapters → models/utils`. Adaptörler
asla servisleri import etmez. Exporter'lar G/Ç içermeyen saf fonksiyonlardır —
yazma işini `ExportService` üstlenir. Her şeyin çevrimdışı test edilebilir
olmasının nedeni budur: her adaptör enjekte edilebilir bir factory/runner alır
(`ExtractorFactory`, `ProcessRunner`, `model_factory`, `cuda_detector`).

---

## 3. Yapılandırmanın çözümlenmesi

`Config` ([app/core/config.py](../app/core/config.py)), `extra="forbid"` ile
donmuş bir Pydantic modelidir. `Config.load()` dört kaynağı, önceliği artan
sırayla birleştirir:

1. Model üzerindeki varsayılan alan değerleri.
2. Çalışma dizinindeki `.env` (`dotenv_values` ile; `os.environ`'a enjekte
   edilmez).
3. Kalıcı kullanıcı dosyası — `$CAPTIONFORGE_CONFIG_FILE`, yoksa
   `$XDG_CONFIG_HOME/captionforge/config.json`, yoksa
   `~/.config/captionforge/config.json`.
4. Süreç ortam değişkenleri.

**Bilinmesi gereken ince bir nokta:** her alan **önce** `CAPTIONFORGE_<ALAN>`
olarak, ancak ondan sonra düz `<alan>` adıyla aranır. Kalıcı JSON dosyası düz
adlar kullanır. Dolayısıyla `.env` içindeki bir `CAPTIONFORGE_RETRY_COUNT`,
`captionforge config set` ile yazılmış bir `retry_count` değerini geçersiz kılar.
Bir ayar yok sayılıyor gibi görünüyorsa sebebi neredeyse her zaman budur.

**Hasara dayanıklılık.** Bozuk bir JSON yapılandırması sessizce atlanır.
Birleştirilmiş değerler bir bütün olarak doğrulamayı geçemezse `load()` alan alan
yeniden dener ve yalnızca tek başına geçerli olanları tutar — bayatlamış tek bir
değer bütün komutları kullanılamaz hâle getiremez. Yalnızca tümüyle başarısızlık
`ConfigurationError` fırlatır.

`config set` yüklemeden daha katıdır: `Config.parse_setting` tek anahtar/değeri
doğrular, bilinmeyen anahtarı veya geçersiz değeri doğrudan reddeder; ardından
model bütün olarak yeniden doğrulanır ve atomik biçimde yazılır.

Pratikte önem taşıyan doğrulama kuralları:

| Ayar | Kural |
|---|---|
| `whisper_device` | `auto` \| `cpu` \| `cuda` |
| `whisper_compute_type` | `auto`, `default`, `int8`, `int8_float16`, `int8_float32`, `int16`, `float16`, `float32`, `bfloat16` |
| `maximum_subtitle_lines` | yalnızca 1 veya 2 |
| `maximum_subtitle_duration` | `minimum_subtitle_duration` değerinden küçük olamaz (alanlar arası doğrulayıcı) |
| `default_output_formats` | virgüllü metin veya tuple; boş olamaz ve srt/vtt/txt/json/docx alt kümesi olmalı |
| `whisper_language`, `whisper_model_download_directory` | boş metin `None`'a (tanımsız) çevrilir |
| `retry_count` | 1–10 |

---

## 4. Komutlar ve tetikledikleri

| Komut | Ağ | Dosya yazar | Ana yol |
|---|---|---|---|
| `version` | hayır | hayır | sabiti basar |
| `config show/set/reset` | hayır | yalnızca kullanıcı yapılandırması | `Config` |
| `doctor` | hayır | test için output/temp dizinlerini oluşturur | yerel kontroller |
| `inspect` | yalnızca meta veri | hayır | `VideoService.inspect` |
| `extract` | meta veri + altyazı izi | evet | altyazı → ayrıştır → son işleme → dışa aktar |
| `transcribe` | meta veri + (altyazı **veya** ses) | evet | altyazı öncelikli, Whisper yedekli |
| `prepare-audio` | meta veri + ses | yalnızca WAV | `AudioService.prepare` |
| `clean` | yok | evet | yerel dosya → ayrıştır → son işleme → dışa aktar |

Her komut `_run_with_config` üzerinden geçer ve aynı dört işi yapar:
yapılandırmayı yükle, loglamayı ayarla, eylemi çalıştır, istisnaları kısa bir
kullanıcı mesajı ile bir çıkış koduna çevir. Ham `yt-dlp`, FFmpeg, CUDA veya
Python metni asla stdout/stderr'e ulaşmaz — log dosyasına gider.

### Çıkış kodları

| Kod | Anlamı |
|---|---|
| 0 | başarılı |
| 1 | genel hata (beklenmeyen istisnalar dâhil) |
| 2 | geçersiz veya desteklenmeyen YouTube URL'si |
| 3 | video erişilemez veya canlı yayın |
| 4 | meta veri alma hatası |
| 130 | `KeyboardInterrupt` |

---

## 5. İnceleme yolu (YouTube'a dokunan her şeyin ortak yolu)

1. **`extract_youtube_video_id`** ([app/utils/url_utils.py](../app/utils/url_utils.py))
   — saf ayrıştırma, ağ yok. `youtube.com`, `www`, `m`, `music` ve `youtu.be`
   alan adlarını; `/watch?v=`, `/shorts/`, `/embed/`, `/v/` ve düz
   `youtu.be/<id>` biçimlerini kabul eder. `/playlist`, `/channel`, `/user`,
   `/c/`, `/@`, `/results`, `/feed` yollarını `UnsupportedYouTubeUrlError` ile,
   diğer her şeyi `InvalidYouTubeUrlError` ile reddeder. Kimlik
   `^[A-Za-z0-9_-]{11}$` desenine uymalıdır. Şema yoksa eklenir.
2. **Kanonikleştirme** — ayrıştırılan kimlikten
   `https://www.youtube.com/watch?v=<id>` yeniden kurulur. İzleme parametreleri,
   `si=`, oynatma listesi bağlamı ve zaman damgaları yt-dlp URL'yi görmeden önce
   atılır.
3. **`YtDlpAdapter.inspect`** — `skip_download`, `noplaylist`, `quiet` ile tek
   bir `extract_info(download=False)` çağrısı. `retry_call` ile sarmalanmıştır.
4. **Hata çevirisi** — `DownloadError` metni `PrivateVideoError`,
   `VideoUnavailableError` (kaldırılmış / üyelere özel / yaş kısıtlı / bölge
   engelli) veya `MetadataRetrievalError` ile eşleştirilir.
5. **Eşleme** — ham sözlük → donmuş `VideoMetadata`. Altyazı sözlükleri
   (`subtitles`, `automatic_captions`) → normalize dil koduna göre sıralanmış
   `SubtitleTrack` demetleri; ayrıştırılamayan dil kodları elenir.
6. **`VideoService` kontrolleri** — `is_live` veya `live_status in {is_live,
   is_upcoming}` → `LiveStreamNotSupportedError`; `availability in {private,
   subscriber_only, premium_only}` → `VideoUnavailableError`.
7. **Seçim** — `SubtitleService.discover`'a devredilir.

### İz seçim kuralları

`SubtitleService.select_track` dört kademeyi katı sırayla uygular ve boş
olmayan ilk kademede durur:

1. Manuel iz, tam normalize eşleşme (`ar-EG` == `ar-EG`)
2. Manuel iz, aynı temel dil (`ar-EG` → herhangi bir `ar*`)
3. Otomatik iz, tam eşleşme
4. Otomatik iz, aynı temel dil

Bir kademe içinde eşitlik durumunda önce düz temel kod (`ar-EG` yerine `ar`),
sonra alfabetik sıra tercih edilir. Dil kodları önce normalize edilir
([app/utils/language_utils.py](../app/utils/language_utils.py)): `_` → `-`, dil
küçük harfe, bölge büyük harfe, yazı sistemi baş harfi büyük — `AR_eg`, `ar-EG`
olur. Bozuk kodlar hata fırlatmak yerine `None` döner ve atlanır.

---

## 6. `extract` — altyazıdan dosyaya

```
incele → iz seç → yalnızca o izi indir → ayrıştır → son işleme → dışa aktar
```

Altyazı indirmesi bir `TemporaryDirectory` içinde, `subtitlesformat: "vtt/best"`
ve `subtitleslangs: [track.language_code]` ile yapılır; `track.is_automatic`
değerine göre `writesubtitles` / `writeautomaticsub` seçeneklerinden tam olarak
biri açılır. Oluşan dosya `utf-8-sig` (BOM toleranslı) ile okunur ve bir
`RawSubtitle` döndürülür.

**Ayrıştırma** iki biçimi ele alır:

- `vtt` / `srt` — satır taramasıyla `BAŞLANGIÇ --> BİTİŞ` bulunur (zaman
  damgalarından sonraki ek cue ayarları yok sayılır), ardından gelen boş
  olmayan tüm satırlar metin bloğunu oluşturur. Zaman damgaları
  `parse_timestamp`'ten geçer; saat kısmı isteğe bağlıdır, milisaniye ayıracı
  olarak hem `,` hem `.` kabul edilir, 59'dan büyük dakika/saniye reddedilir.
- `json3` — YouTube'un JSON altyazı formatı; `events[].segs[].utf8` birleştirilir,
  zamanlama `tStartMs` + `dDurationMs` ile kurulur.

Bunların dışındaki her şey `SubtitleParseError` fırlatır.

`--no-postprocess` ile bunun yerine asgari bir temizlik yolu çalışır
(`_clean_segments`): işaretleme temizlenir, ardışık birebir tekrarlar atılır,
çakışmalar kırpılır — ancak birleştirme, bölme veya yeniden satırlama yapılmaz.
Kaynak segmentasyonunun birebir korunması gerektiğinde bunu kullanın.

---

## 7. `transcribe` — altyazı öncelikli, Whisper yedekli

İki kaynak arasındaki kararı veren tek yer `TranscriptionService.process`'tir.

```
incele (%5)
  ├── iz bulundu ve --force yok  → altyazıyı indir (%25) → dışa aktar (%85) → bitti
  └── aksi hâlde
        sesi hazırla (%10 → %25)
          ├── iş dizini oluştur + disk kontrolü
          ├── yt-dlp bestaudio indirme
          └── ffmpeg → mono 16 kHz PCM WAV
        modeli yükle (%30)
        transkribe et (%40 → %85)
        son işleme (%88)
        dışa aktar (%92)
        bitti (%100)
```

İlerleme yüzdeleri gerçek ve belirlenimcidir — ses alt ilerlemesi
`10 + yüzde * 0.15` ile yeniden eşlenir, Whisper'ın segment döngüsü geçen ses
süresini 40–85 aralığına taşır (`40 + min(45, bitiş/süre*45)`); süre
bilinmiyorsa `min(84, 40 + segment_sırası)` şeklinde yavaş bir sayaca düşer.

### Ses hazırlama ([app/services/audio_service.py](../app/services/audio_service.py))

- UUID'li bir `Job` oluşturulur; iş alanı
  `<temp_directory>/captionforge-<uuid>` olup `0o700` modunda yaratılır ve bir
  yazma testiyle denenir.
- Gereken disk alanı `max(minimum_free_disk_bytes, süre_saniye * 64000)` —
  yaklaşık PCM boyutunun iki katı, sıkıştırılmış indirme ile WAV çıktısını
  birlikte karşılamak için. Süre bilinmiyorsa 600 sn varsayılır.
- `yt-dlp`, `format: "bestaudio"` ile. "requested format is not available"
  hatası `AudioFormatUnavailableError`'a, diğer her şey yeniden denenebilir
  `AudioDownloadError`'a çevrilir.
- FFmpeg asla shell üzerinden değil, argüman listesiyle çağrılır:
  `-y -i <kaynak> -vn -acodec pcm_s16le -ar 16000 -ac 1 <hedef>`. Sonrasında
  çıktının var olduğu ve boş olmadığı doğrulanır. `audio_format` yalnızca dosya
  uzantısını değiştirir — codec her zaman `pcm_s16le`'dir, Whisper'ın istediği
  budur.
- Saklama istenmediğinde başarı sonrası WAV
  `<temp>/captionforge-<uuid>.wav` konumuna taşınır ve iş dizini silinir.
  Saklama istendiğinde her şey iş dizininde kalır.
- `KeyboardInterrupt` → iş `CANCELLED` + iş alanı silinir +
  `ProcessingInterruptedError`. Herhangi bir `CaptionForgeError` → iş `FAILED` +
  iş alanı silinir.

Tek başına `prepare-audio`, eşleşen bir altyazı varsa `--force` olmadan
çalışmayı reddeder — normal akışın parçası değil, bir tanılama komutudur.
`transcribe` içinden her zaman `force=True` (karar zaten verilmiştir) ve
`keep_temp=True` (temizliği `finally` bloğunda çağıran üstlenir) ile çağrılır.

### Whisper adaptörü ([app/adapters/whisper_adapter.py](../app/adapters/whisper_adapter.py))

- `faster_whisper` **tembel** biçimde, `importlib` ile, transkripsiyon anında
  import edilir. Uygulamanın geri kalanı o paket kurulu olmadan çalışır; eksik
  paket `WhisperNotInstalledError` olarak görünür.
- Cihaz: `auto` → `ctranslate2.get_cuda_device_count()` sıfırdan farklıysa
  `cuda`, değilse `cpu`. GPU yokken açıkça `cuda` istenirse sessizce düşmek
  yerine `CudaUnavailableError` fırlatılır.
- Hesaplama tipi: `auto` → CUDA'da `float16`, CPU'da `int8`.
- VAD varsayılan olarak açıktır, `min_silence_duration_ms=500`.
- İptal, model yüklemesinden önce ve üretilen her segmentte kontrol edilir.
- Motor nesneleri dışarı sızmaz: segmentler döngü içinde donmuş
  `TranscriptionSegment` modellerine çevrilir, üreteç kapatılır ve GPU belleğini
  hızlıca bırakmak için `finally` içinde `gc.collect()` çalışır.
- Hata çevirisi istisna mesajı üzerinden metin eşlemesiyle yapılır: "out of
  memory" → `GpuMemoryError`; "compute type"/"quantization" →
  `InvalidComputeTypeError`; yükleme sırasında "invalid model"/"model not
  found"/"repository not found" → `UnsupportedModelError`; diğer yükleme
  hataları → yeniden denenebilir `ModelLoadError`; çalışma anı hataları →
  `AudioTranscriptionError`.
- `confidence`, 0–1 aralığına sıkıştırılmış `exp(avg_logprob)` değeridir.
  Sıralama için yararlı bir skordur; kasten kalibre edilmiş bir olasılık olarak
  sunulmaz.

Kullanılabilir segment sayısı sıfırsa `EmptyTranscriptionError` fırlatılır —
boş dosya asla yazılmaz.

---

## 8. Son işleme — çıktıyı asıl şekillendiren kısım

`PostProcessingService._process`
([app/services/postprocessing_service.py](../app/services/postprocessing_service.py))
indirilen altyazılar ile Whisper çıktısı için aynı altı aşamayı, bu sabit sırayla
çalıştırır:

**1. Metin temizliği** (`clean_caption_text`)
HTML varlıkları çözülür, sıfır genişlikli boşluklar silinir, `<...>` etiketleri
kaldırılır, boşluklar sadeleştirilir. *Tamamen* `[...]` veya `(...)` olan bir cue
atılır. Noktalama boşlukları normalize edilir — `، ؛ ؟ , . ! ? : ; ٪ %`
işaretlerinden önceki boşluk silinir, sonrasına boşluk eklenir; **ancak**
rakamlar arasında bu yapılmaz, böylece `3.14` ve `1,000` korunur. Hareke
temizliği, elif/ya normalizasyonu ve Arap-Hint rakam dönüşümü **varsayılan olarak
kapalıdır**; siz istemedikçe konuşmacının harflerine dokunulmaz. Ardından ≥2
kelimelik bitişik tekrar eden ifadeler sadeleştirilir
(`_collapse_repeated_phrase`). Tanınan bir sessizlik cue'su (music/applause/
silence/موسيقى/تصفيق/صمت) yalnızca `no_speech_probability ≥ 0.9` ise atılır.

**2. Tekrar temizliği**
Yalnızca bir önceki segmentle, harf büyüklüğü yok sayılmış temizlenmiş bir
anahtar üzerinden karşılaştırılır. Birebir aynıysa → önceki segmentin bitişi
uzatılır ve yenisi atılır. Aksi hâlde
`SequenceMatcher.ratio() ≥ duplicate_detection_threshold` (0.9) veya tespit
edilen bir baş-ifade örtüşmesi düzeltmeyi tetikler: tekrar eden baş ifade
mevcut metinden kırpılır ya da — mevcut metin daha uzun değilse — önceki
segmente soğurulur. YouTube otomatik altyazılarının tipik kayan tekrar
artefaktını ortadan kaldıran mekanizma budur.

**3. Zamanlama onarımı (1. geçiş, asgari süre uygulanmadan)**
Negatif başlangıçlar 0'a çekilir; bir çakışma ya önceki segmenti kısaltır ya da
mevcut başlangıcı ileri iter; süreler `maximum_subtitle_duration` ile
sınırlanır.

**4. Kısa segmentleri birleştirme**
İki komşu, *tüm* şu koşullar sağlanınca birleşir: aradaki boşluk ≤
`subtitle_merge_threshold` (1.0 sn), birleşik uzunluk ≤
`maximum_characters_per_line × maximum_subtitle_lines` (varsayılan 84), en az
bir taraf `minimum_subtitle_duration`'dan kısa ya da tek kelimelik, ve önceki
metin zaten bir cümle sonu (`. ! ? ؟ ؛ …`) ile bitmiyor.

**5. Uzun segmentleri bölme**
Hem karakter bütçesine hem `maximum_subtitle_duration`'a göre bölünür; sınırın
yarısı geçildikten sonra cümle sonları tercih edilir. Yeni zamanlamalar
`distribute_duration` ile üretilir; aralık, parça uzunluğuyla orantılı olarak
kümülatif sınırlarla paylaştırılır — böylece parçalar bitişik olur, kayan nokta
kaynaklı boşluk veya çakışma oluşmaz.

**6. Zamanlama onarımı (2. geçiş, asgari süre uygulanır)** ve ardından **satır
sarma**
`_wrap`, iki yarıyı en dengeli biçimde bölen kelime sınırını bulur ve orada
böler — ancak yalnızca *her iki* yarı da `maximum_characters_per_line` içine
sığıyorsa. Aksi hâlde satır kötü bölünmektense uzun bırakılır.

`--no-postprocess`, `extract` ve `transcribe` için tüm bunları atlar. `clean`
komutu ise elinizde zaten olan bir dosyaya uygulanan son işlemedir.

---

## 9. Dışa aktarım ve dosya güvenliği

`ExportService.export` ([app/services/export_service.py](../app/services/export_service.py)):

1. Formatlar küçük harfe çevrilir ve sıra korunarak tekilleştirilir
   (`dict.fromkeys`); bilinmeyen formatlar hiçbir şey yazılmadan hata fırlatır.
2. Çıktı dizini oluşturulur ve `W_OK` için kontrol edilir.
3. Dosya adı gövdesi `sanitize_filename(video.title)`'dan gelir: NFC normalize
   edilir, `<>:"/\|?*` ve kontrol karakterleri `_` ile değiştirilir, boşluklar
   sadeleştirilir, baştaki/sondaki boşluk ve noktalar kırpılır, 180 karaktere
   kısaltılır. Sonuç boşsa veya Windows'un ayrılmış adlarından biriyse (`CON`,
   `NUL`, `COM1`…) video kimliğine düşülür. Arapça, Türkçe ve diğer Unicode
   karakterler kasten korunur.
4. **Tüm** hedef yolların varlığı, render işleminden *önce* kontrol edilir.
   `--overwrite` yoksa komut hiçbir şey yazmadan durur.
5. İçerikler bellekte üretilir, toplam UTF-8 bayt boyutu boş disk alanına karşı
   kontrol edilir, sonra her dosya `atomic_write_text` ile yazılır: aynı dizinde
   `mkstemp` → yaz → `flush` → `os.fsync` → `Path.replace`. Bir okuyucu asla
   yarım dosya görmez ve bir çökme mevcut dosyayı bozamaz.
6. Çok formatlı bir dışa aktarımda sonraki bir dosya başarısız olursa, aynı
   çağrıda yeni oluşturulmuş dosyalar silinir — yarım kalmış set bırakılmaz.

Render'lar ([app/exporters/](../app/exporters/)):

- **SRT** — 1'den başlayan indeks, `HH:MM:SS,mmm`, dışa aktarımda yeniden
  numaralandırılır.
- **VTT** — `WEBVTT` başlığı, `HH:MM:SS.mmm`, cue kimliği yok.
- **TXT** — satır başına bir segment; `--timestamped-txt` başa
  `[HH:MM:SS.mmm]\t` ekler.
- **JSON** — tam `VideoMetadata`, seçilen dil, altyazı kaynak tipi ve
  `confidence` ile `no_speech_probability` dâhil tüm segment alanları.
  `ensure_ascii=False` olduğu için Arapça dosyada okunabilir kalır.

---

## 10. Hatalar, yeniden denemeler, loglama

**Hiyerarşi.** Beklenen her şey `CaptionForgeError`'dan türer; bu sınıf
kullanıcıya gösterilen bir `message` ile teknik bir `details` taşır. Temel sınıf
`retryable = False` sunar; bunu `True` yapan yalnızca dört tip vardır:

- `MetadataRetrievalError`
- `SubtitleDownloadError`
- `AudioDownloadError` (ve alt sınıfı `AudioFormatUnavailableError`)
- `ModelLoadError` (ve alt sınıfı `UnsupportedModelError`)

**Yeniden deneme.** `retry_call` ([app/core/retry.py](../app/core/retry.py)) bir
işlemi yalnızca *çevrilmiş* istisna yeniden denenebilir olduğunda tekrarlar.
Geçersiz URL'ler, erişilemeyen videolar, eksik FFmpeg, bozuk zaman damgaları ve
geçersiz çıktı yolları ilk denemede başarısız olur. Yeniden denemeler, deneme
numarası ve teknik neden ile loglanır. `retry_count` ve `retry_delay_seconds`
yapılandırılabilir; gecikme sabittir, üstel değildir.

`UnsupportedModelError`'ın `ModelLoadError`'dan `retryable = True` miras aldığını
unutmayın: gerçekten hatalı bir model adı da başarısız olmadan önce `retry_count`
kez denenir.

**Loglama.** `configure_logging`, Loguru'nun varsayılan handler'ını kaldırır ve
**yalnızca bir dosya sink'i** ekler — `logs/captionforge_YYYY-MM-DD.log`, 10
MB'ta döner, 14 gün saklanır, UTF-8, `enqueue=True`; `backtrace` ve `diagnose`
kapalıdır, böylece yerel değişkenler dosyaya sızmaz. Logger'dan hiçbir şey
terminale ulaşmaz; kullanıcıya gösterilen metin ayrıca Rich ile basılır.
Loglanan bilgiler arasında iş kimlikleri, aşamalar, seçilen yöntem,
model/cihaz/hesaplama tercihleri, yeniden deneme sayıları, çıktı yolları ve
süreler bulunur.

**İptal.** `Ctrl-C`, `KeyboardInterrupt` olarak yayılır,
`ProcessingInterruptedError` / `TranscriptionCancelledError`'a çevrilir, iş
`CANCELLED` olarak işaretlenir, saklama istenmedikçe geçici ve yarım dosyalar
silinir ve süreç 130 ile çıkar.

---

## 11. `Job` modeli

`Job` ([app/models/job.py](../app/models/job.py)) tek bir çalıştırmanın süreç içi
kaydıdır: UUID, kaynak URL, dil, formatlar, durum, zaman damgaları, iş alanı ve
ses yolları, ilerleme, mevcut aşama, hata aşaması ve toplam süre. `transition()`
ilk pending-dışı duruma geçişte `started_at`'i damgalar; `COMPLETED`/`FAILED`/
`CANCELLED` durumunda `finished_at`'i damgalar, `duration_seconds`'ı hesaplar ve
başarısızlıkta `failure_stage`'i kaydeder. Çalıştırmalar arasında kalıcı değildir
— loglara tutarlı ve ilişkilendirilebilir bir biçim vermek için vardır.

---

## 12. Geliştirme

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest --cov=app --cov-report=term-missing
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check app tests
.venv/bin/python -m mypy app
```

Varsayılan test paketi tamamen çevrimdışıdır —
`addopts = "-ra -m 'not integration'"`. Canlı testler isteğe bağlıdır ve
`CAPTIONFORGE_INTEGRATION_VIDEO_URL` (YouTube) veya
`CAPTIONFORGE_INTEGRATION_AUDIO` (gerçek Whisper) gerektirir; `pytest -m
integration` ile çalıştırılır.

Çevrimdışı test mümkündür çünkü her dış sınır enjekte edilebilir:
`YtDlpAdapter(extractor_factory=…)`, `FFmpegAdapter(runner=…)`,
`WhisperAdapter(model_factory=…, cuda_detector=…)` ve `retry_call(sleep=…)`.

---

## 13. Genişletme

| Amaç | Dokunulacak yer |
|---|---|
| Yeni çıktı formatı | `app/exporters/` içine render fonksiyonu ekle, `ExportService.export` içinde kaydet, `SUPPORTED_OUTPUT_FORMATS`'a ekle |
| Farklı transkripsiyon motoru | `TranscriptionResult` döndüren yeni bir adaptör; `TranscriptionService` değişmez |
| Başka bir video kaynağı | aynı `inspect`/`download_subtitle`/`download_audio` biçimine sahip yeni adaptör |
| Farklı altyazı stil kuralları | `PostProcessingService` ve ilgili `Config` alanları |
| Yeni dil görünen adı | `app/utils/language_utils.py` içindeki `_LANGUAGE_NAMES` |

---

## 14. Bilinen sınırlar ve pürüzler

Kasıtlı sınırlar: yalnızca tekil, canlı olmayan videolar; oynatma listesi,
kanal, canlı yayın, çeviri, konuşmacı ayrıştırma, çerez/kimlikli erişim veya GUI
yok; agresif yazım/dilbilgisi yeniden yazımı yok.

Mevcut koddaki, bilinmesinde fayda olan pürüzler:

- `prepare-audio` hâlâ "Transcription will be implemented in Phase 5" yazıyor ve
  `extract`, eşleşen altyazı bulunmadığında transkripsiyon yedeğinin "will be
  added in a later phase" olduğunu söylüyor. İkisi de bayat — `transcribe` bunu
  bugün zaten yapıyor.
- `FFmpegAdapter.build_conversion_command` içinde
  `codec = "pcm_s16le" if … else "pcm_s16le"` şeklinde bir dal var;
  `audio_format` yalnızca dosya uzantısını etkiliyor.
- `configure_logging`'in docstring'i konsol ve dosya handler'ından söz ediyor,
  ancak yalnızca dosya handler'ı kaydediliyor.
- `clean` komutu `ExportService`'i yeniden kullanabilmek için `localclean1`
  sentetik kimliğiyle bir yer tutucu `VideoMetadata` kuruyor; dosya sonrasında
  istenen hedefe yeniden adlandırılıyor.
