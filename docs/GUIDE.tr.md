# CaptionForge — Uygulama Nasıl Çalışır

Uygulamanın iç işleyişine dair çalışan bir rehber: hangi katman ne yapıyor, her
komutta ne oluyor ve çıktıyı hangi kurallar belirliyor. Sürüm 0.9.0, Python
3.12+.

---

## 1. Uygulama gerçekte ne yapıyor

CaptionForge tek bir YouTube video URL'sini veya bu bilgisayardaki bir video ya
da ses dosyasını altyazı/transkript dosyalarına (`srt`, `vtt`, `txt`, `json`,
`docx`) dönüştürür. İki metin kaynağı vardır ve her zaman
ucuz olanı tercih eder:

1. **Mevcut YouTube altyazıları** — yalnızca altyazı izi indirilir, medya
   indirilmez.
2. **Transkripsiyon** — yalnızca istenen dilde eşleşen bir altyazı yoksa
   (veya `--force` verildiyse). Sadece ses akışı indirilir, mono 16 kHz PCM WAV'a
   dönüştürülür ve yerel makinede `faster-whisper`'a verilir. Bu bilgisayardaki
   bir dosyanın YouTube altyazısı olmadığından her zaman bu yolu izler; sesi
   indirilmez, dosya bulunduğu yerde dönüştürülür. `deepgram` motoru seçildiyse
   ses bunun yerine mono Opus olarak sıkıştırılıp Deepgram'a yüklenir ve
   transkript oradan geri gelir (bölüm 7).

Hiçbir yolda video akışı indirilmez. YouTube (`yt-dlp` üzerinden), ilk model
indirmesinde Whisper model sunucusu, CaptionForge kendi paketlerinin yeni
sürümlerini ararken PyPI ve yalnızca o motor seçilip bir transkripsiyon gerçekten
çalıştığında ya da yapıştırılan bir anahtar kontrol edilirken Deepgram dışında
hiçbir servise ağ çağrısı yapılmaz. Hiçbir şey sorulmadan kurulmaz.

---

## 2. Katman haritası

```
app/
├── main.py              giriş noktası → app.interfaces.cli:app  ("captionforge" konsol betiği)
├── interfaces/cli.py    Typer komutları, Rich çıktısı, hata→çıkış kodu eşlemesi
├── interfaces/desktop/  Qt penceresi: sayfanın bölümleri, yerel bileşenler olarak
├── interfaces/launcher.py  uygulamayı başlatan menü girdisini yazar
├── services/            orkestrasyon; iş akışının kararlaştırıldığı tek yer
│   ├── video_service.py         URL veya dosya → doğrulama + kontroller + keşif
│   ├── subtitle_service.py      iz seçimi, altyazı ayrıştırma, asgari temizlik
│   ├── audio_service.py         iş alanı, ses indirme, FFmpeg dönüştürme
│   ├── transcription_service.py altyazı-öncelikli akış, Whisper/Deepgram yedeği, dışa aktarım
│   ├── postprocessing_service.py  zamanlama + metin normalizasyonu (tüm kaynaklar için ortak)
│   └── export_service.py        format doğrulama, dosya adı, atomik çok formatlı yazma
├── adapters/            dış dünyayla konuşan her şey
│   ├── ytdlp_adapter.py    meta veri, altyazı indirme, ses indirme, hata çevirisi
│   ├── ffmpeg_adapter.py   alt süreç (shell yok), dönüştürme, FFprobe, hata çevirisi
│   ├── whisper_adapter.py  tembel faster-whisper importu, cihaz/hesaplama seçimi
│   └── deepgram_adapter.py Deepgram'a akışlı yükleme, yanıt → TranscriptionResult
├── models/              donmuş (frozen) Pydantic sözleşmeleri (VideoMetadata, SubtitleTrack, …)
├── exporters/           saf render fonksiyonları: segmentler → metin
├── utils/               saf yardımcılar (URL, yerel yol, zaman, dil, dosya adı, Arapça metin)
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
| `transcription_engine` | `whisper` (varsayılan) \| `deepgram` |
| `deepgram_model` | herhangi bir Deepgram model adı; varsayılan `nova-3` |
| `whisper_device` | `auto` \| `cpu` \| `cuda` |
| `whisper_compute_type` | `auto`, `default`, `int8`, `int8_float16`, `int8_float32`, `int16`, `float16`, `float32`, `bfloat16` |
| `maximum_subtitle_lines` | yalnızca 1 veya 2 |
| `maximum_subtitle_duration` | `minimum_subtitle_duration` değerinden küçük olamaz (alanlar arası doğrulayıcı) |
| `default_output_formats` | virgüllü metin veya tuple; boş olamaz ve srt/vtt/txt/json/docx alt kümesi olmalı |
| `whisper_language`, `whisper_model_download_directory` | boş metin `None`'a (tanımsız) çevrilir |
| `retry_count` | 1–10 |
| `check_for_updates` | `true` (varsayılan) sayfa açıldığında ve YouTube yt-dlp'yi reddettiğinde güncelleme arar; `false` yalnızca `captionforge update` ile arar. Kurmadan önce her zaman sorar. |

---

## 4. Komutlar ve tetikledikleri

| Komut | Ağ | Dosya yazar | Ana yol |
|---|---|---|---|
| `version` | hayır | hayır | sabiti basar |
| `config show/set/reset` | hayır | yalnızca kullanıcı yapılandırması | `Config` |
| `doctor` | hayır | test için output/temp dizinlerini oluşturur | yerel kontroller |
| `inspect` | yalnızca meta veri | hayır | `VideoService.inspect` |
| `extract` | meta veri + altyazı izi | evet | altyazı → ayrıştır → son işleme → dışa aktar |
| `transcribe` | meta veri + (altyazı **veya** ses); `--engine deepgram` ile Deepgram'a yükleme | evet | altyazı öncelikli, Whisper veya Deepgram yedekli |
| `prepare-audio` | meta veri + ses | yalnızca WAV | `AudioService.prepare` |
| `clean` | yok | evet | yerel dosya → ayrıştır → son işleme → dışa aktar |
| `web` | sayfanın istediği kadar | evet | `127.0.0.1` üzerinde FastAPI |
| `desktop` | pencerenin istediği kadar | evet | aynı servisler üzerinde bir Qt penceresi |
| `install-desktop` | hayır | bir masaüstü girdisi ve bir simge | `interfaces/launcher.py` |
| `update` | PyPI | seçtiğiniz paketler | `PackageUpdater.check` → sor → `install` |
| `deepgram-key` | anahtarı kontrol için Deepgram | `deepgram.key` (mod 0600) | `check_and_save_key`; `--forget` siler |

Bağlantı yerine bu bilgisayardaki bir dosya verildiğinde `inspect`, `transcribe`
ve `prepare-audio` (ilk model indirmesi dışında) hiç ağ çağrısı yapmaz; `extract`
ve `download` ise dosyayı, yerine ne çalıştırılacağını söyleyen bir cümleyle
reddeder.

Her komut `_run_with_config` üzerinden geçer ve aynı dört işi yapar:
yapılandırmayı yükle, loglamayı ayarla, eylemi çalıştır, istisnaları kısa bir
kullanıcı mesajı ile bir çıkış koduna çevir. Eylem `ExtractorRefusedError` ile
başarısız olursa ve terminalde biri varsa, önce daha yeni bir yt-dlp önerir;
kurulduysa eylemi bir kez daha çalıştırır. Ham `yt-dlp`, FFmpeg, CUDA veya
Python metni asla stdout/stderr'e ulaşmaz — log dosyasına gider.

### Masaüstü kipi

`captionforge desktop` ([app/interfaces/desktop/](../app/interfaces/desktop/))
web sunucusunun çağırdığı servisleri aynı süreç içinde çağıran bir Qt (PySide6)
penceresi açar: arama için `create_video_service().inspect_all`, altyazı ve
indirmeler için `JobRegistry`, güncellemeler için `PackageUpdater` ve web
tercihleri dosyası; böylece iki arayüz de aynı seçimleri hatırlar. Arada HTTP
sunucusu, port ya da token yoktur.

- `theme.py`, `app.css`'i yeniden ifade eder: açık ve koyu için aynı
  değişkenler, Qt stil sayfası olarak aynı kurallar; sistem renk şemasını
  değiştirdiğinde anında geçiş yapar. Qt stil sayfalarının ifade edemediklerini
  (`letter-spacing`, büyük harf, `opacity`, `:has()`, 1.55 satır yüksekliği)
  bileşenler kendileri yapar.
- `widgets.py` sayfanın görsel sözlüğüdür: çipler, seçenek kartları, ilerleme
  çubuğu, dosya satırları, Options açılır bölümü.
- `window.py` bölümleri sayfadaki sırayla kurar ve `app.js`'i izler: işi her
  700 ms'de bir yoklar, son değişiklikten 600 ms sonra seçimleri kaydeder,
  YouTube reddettiğinde daha yeni bir yt-dlp önerir. Yavaş çağrılar ayrı bir iş
  parçacığında çalışır ve sonucu pencerenin iş parçacığına iletir
  (`background.py`). Choose file sistemin dosya seçicisini açar; pencerenin
  herhangi bir yerine bırakılan bir dosya da alınır (metin alanları bırakmayı
  reddeder, böylece dosyayı yutamazlar). Her iki durumda da yol alana yazılır ve
  dosya kopyalanmadan bulunduğu yerde incelenir.
- `application.py` sürecin kendisini yönetir:

1. **Tek örnek.** `$XDG_RUNTIME_DIR` içindeki bir `QLockFile` hangi kopyanın
   ilk olduğunu belirler; ilk kopya yanındaki, yalnızca kullanıcının
   erişebildiği yerel bir soketi dinler. Sonraki bir çalıştırma bu sokete
   seslenir, açık pencere öne gelir ve sonraki çalıştırma çıkar. Kilit sahibinin
   PID'ine bağlıdır; bu yüzden bir çökme bir sonraki açılışı asla engellemez.
2. **Masaüstüne göre adlandırılmış.** `QApplication`, argv[0] `captionforge`
   olarak başlatılır; X11 bunu WM_CLASS örnek adı olarak kullanır.
   `desktopFileName` de `captionforge`'dur, yani Wayland uygulama kimliği.
   Girdideki `StartupWMClass=captionforge` sayesinde GNOME pencereyi
   CaptionForge simgesinin altında gösterir.
3. **Bir ömür.** `JobRegistry` içinde bitmemiş iş varken pencereyi kapatmak onu
   yalnızca gizler; uygulama son iş bittiğinde sona erer. `SIGTERM` (oturum
   kapatma) ve terminaldeki Ctrl+C onu hemen sonlandırır.

Komut satırı yalnızca Qt'yi içe aktarmayan `app/interfaces/desktop/__init__.py`
dosyasını yükler; böylece `[desktop]` eki kurulu olmadan da geri kalan her şey
çalışır.

`install-desktop` ([app/interfaces/launcher.py](../app/interfaces/launcher.py))
`<sys.executable> -m app desktop` çalıştıran bir girdi yazar ve `Path=` alanını
kurulumun yapıldığı klasöre ayarlar; göreli `default_output_folder`, `temp/` ve
`logs/` yollarının komut satırındakiyle aynı yere düşmesini sağlayan budur.
Girdi ayrıca `StartupWMClass=captionforge` ve `SingleMainWindow=true` bildirir.

### Çıkış kodları

| Kod | Anlamı |
|---|---|
| 0 | başarılı |
| 1 | genel hata (beklenmeyen istisnalar dâhil) |
| 2 | geçersiz veya desteklenmeyen YouTube URL'si; ya da bulunamayan, okunamayan veya sessiz bir dosya |
| 3 | video erişilemez veya canlı yayın |
| 4 | meta veri alma hatası |
| 130 | `KeyboardInterrupt` |

---

## 5. İnceleme yolu (bağlantı veya dosya alan her şeyin ortak yolu)

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

### Bu bilgisayardaki dosyalar

`VideoService.inspect_all` önce `local_media_path`'e
([app/utils/local_media.py](../app/utils/local_media.py)) girdinin bir dosyayı
gösterip göstermediğini sorar. Sistemlerin yolları kopyaladığı biçimlerin hepsini
kabul eder: düz yol, tırnak içinde yol, `~/…`, `file://` URI'leri (yüzde
kodlaması çözülür, `localhost` kabul edilir, başka bir sunucu reddedilir),
Windows sürücü yolları, `./` ve `../`, `LOCAL_MEDIA_EXTENSIONS` içindeki bir
medya uzantısını taşıyan çıplak bir ad ve var olan bir dosyanın adı. `file:`
dışında bir URL şeması taşıyan her şey bağlantıdır. Hiçbir şeyi göstermeyen bir
yol, bozuk bir YouTube URL'si olarak değil, bulunamayan bir dosya olarak
bildirilir.

Ardından, yt-dlp hiç devreye girmeden:

1. `resolve_media_file` yolu mutlak yapar (FFprobe çıplak bir `-ad` değerini
   seçenek olarak okur) ve bulunamayan bir yol ya da klasör için
   `LocalFileNotFoundError`, kullanıcının okuyamadığı bir dosya için
   `UnreadableMediaFileError` fırlatır.
2. `FFmpegAdapter.probe`, yapılandırılmış FFmpeg'in yanında bulunan FFprobe'u
   (`/opt/ff/ffmpeg` → `/opt/ff/ffprobe`) `-show_format -show_streams` ile JSON
   çıktısında çalıştırır. Süreyi (kapsayıcınınkini, yoksa en uzun akışınkini) ve
   herhangi bir akışın ses olup olmadığını okur. FFprobe'un reddettiği bir dosya
   `UnreadableMediaFileError`, sesi olmayan bir dosya ise `NoAudioStreamError`
   olur; bu, bir model yüklendikten sonra değil, burada reddedilir.
3. Sonuç, `local_path` dolu ve `video_id` `None` olan bir `VideoMetadata`'dır
   (bir model doğrulayıcısı ikisinden tam olarak birini ister). Başlık dosyanın
   gövde adıdır, böylece `lecture.mp4` `lecture.srt` olarak dışa aktarılır;
   altyazı izi yoktur ve `MediaOptions` boştur, bu da sayfadaki indirme satırını
   gizler.

Bu üç hatanın hepsi `LocalMediaError`'dır: sayfada HTTP 400, terminalde çıkış
kodu 2.

Tarayıcı, seçilen ya da bırakılan bir dosyanın nerede durduğunu sayfaya asla
söylemez; bu yüzden sayfa dosyanın baytlarını `/api/uploads`'a `PUT` eder
([app/interfaces/web/uploads.py](../app/interfaces/web/uploads.py)). Sunucu
bunları `<temp_directory>/uploads/<uuid>/<ad>` konumuna yazar (`0o700` modu,
temizlenmiş ad, önce disk alanı kontrolü) ve bu mutlak yolla yanıt verir. Sayfa
ardından bu yolu yapıştırılmış bir yol gibi inceler. Daha yeni bir kopya, henüz
bitmemiş bir işin hâlâ okuduğu kopya hariç eskilerin yerini alır; sunucu
durduğunda tüm kopyalar silinir, bir çökmeden kalan ve bir günden eski kopyalar
da açılışta süpürülür. Yolu yapıştırmak kopyayı atlar; masaüstü penceresi hiç
kopya yapmaz.

### İz seçim kuralları

`SubtitleService.select_track` dört kademeyi katı sırayla uygular ve boş
olmayan ilk kademede durur:

1. Manuel iz, tam normalize eşleşme (`ar-EG` == `ar-EG`)
2. Manuel iz, aynı temel dil (`ar-EG` → herhangi bir `ar*`)
3. Otomatik iz, tam eşleşme
4. Otomatik iz, aynı temel dil

Makine çevirisi izler, kademeler değerlendirilmeden önce elenir. YouTube,
otomatik transkripsiyonunun çevirilerini ~150 dil için yayımlar; bunlar altyazı
URL'sindeki `tlang=` ile ayırt edilir. Bir transkripsiyonun çevirisi, yerel
transkripsiyondan daha kötüdür; bu yüzden *hiç iz olmaması*nın da altında
sıralanır ve `transcribe`'ın Whisper'a düşmesini sağlar. `--allow-translated`
ile geri açılır; `SubtitleService.translated_matches` CLI'nin bu geri düşüşü
açıklamasını sağlar.

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

Altyazı indirmesi bir `TemporaryDirectory` içinde,
`subtitlesformat: "json3/vtt/best"` ve `subtitleslangs: [track.language_code]`
ile yapılır; `track.is_automatic`
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

**json3 neden önce isteniyor.** YouTube'un otomatik VTT'si kayan (karaoke)
bir formattır: her cue bir önceki satırı tekrarlar ve sonrakini ekler. Ölçülen
bir örnekte aynı iz için VTT 226 cue / 2322 kelime üretirken json3 114 cue / 780
kelime üretti; ayrıca VTT ilk ifadenin gerçek başlangıcını da kaybediyordu
(7.230 sn yerine 5.200 sn). json3 ifade başına tek cue taşır ve onarım
gerektirmez. json3 sunmayan izler için VTT yedek olarak kalır.

`--no-postprocess` ile bunun yerine asgari bir temizlik yolu çalışır
(`_clean_segments`): işaretleme temizlenir, ardışık birebir tekrarlar atılır,
çakışmalar kırpılır — ancak birleştirme, bölme veya yeniden satırlama yapılmaz.
Kaynak segmentasyonunun birebir korunması gerektiğinde bunu kullanın.

---

## 7. `transcribe` — altyazı öncelikli, sonra Whisper veya Deepgram

Kaynaklar ve iki motor arasındaki kararı veren tek yer
`TranscriptionService.process`'tir.

```
incele (%5)
  ├── iz bulundu ve --force yok  → altyazıyı indir (%25) → dışa aktar (%85) → bitti
  └── aksi hâlde
        Deepgram seçili mi? → anahtar yok → DeepgramKeyMissingError, hiçbir indirmeden önce
        sesi hazırla (%10 → %25)
          ├── iş dizini oluştur + disk kontrolü
          ├── yt-dlp bestaudio indirme   (dosyada atlanır)
          └── ffmpeg → mono 16 kHz PCM WAV   (Deepgram: mono Opus, 48 kbps)
        Whisper:  modeli yükle (%30) → transkribe et (%40 → %85)
        Deepgram: yükle (%40 → %75) → bekle (%78) → yanıtı oku (%85)
        son işleme (%88)
        dışa aktar (%92)
        bitti (%100)
```

Motor istekten gelir (`--engine`, sayfanın veya pencerenin **Transcribe with**
çipleri), yoksa `transcription_engine` ayarından. Altyazı izi yeniden kullanılan
bir video hiçbir motora, dolayısıyla anahtara da ihtiyaç duymaz.

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
  `AudioDownloadError`'a çevrilir. Bu bilgisayardaki bir dosya indirmeyi atlar:
  FFmpeg onu yerinde okur; dosya asla taşınmaz, kopyalanmaz veya silinmez.
- FFmpeg asla shell üzerinden değil, argüman listesiyle çağrılır:
  `-y -i <kaynak> -vn -acodec pcm_s16le -ar 16000 -ac 1 <hedef>`. Sonrasında
  çıktının var olduğu ve boş olmadığı doğrulanır. Codec kapsayıcıya göre
  seçilir: `wav` → `pcm_s16le` (Whisper'ın istediği), `ogg` → `libopus`,
  `-b:a 48k -vbr constrained` ile (Deepgram'a gönderilen). Kısıtlı VBR dosyayı
  arayüzlerin gösterdiği boyutun yaklaşık %2 yakınında tutar; Opus'un serbest
  VBR'ı sabit bir tonda %27 aşmıştı.
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

### Deepgram adaptörü ([app/adapters/deepgram_adapter.py](../app/adapters/deepgram_adapter.py))

- Yalnızca standart kütüphane (`urllib`); SDK yok. `model`, `smart_format`,
  `punctuate`, `utterances` ve `language` ya da `detect_language` ile tek bir
  `POST /v1/listen`. "Names and spellings" virgüllerden bölünüp tekrarlanan
  `keyterm` (Nova-3, Flux) veya `keywords` (eski modeller) parametrelerine
  dönüşür; Deepgram 500 tokeni aşan isteği reddettiği için 300 kelimeyle
  sınırlanır.
- Gövde dosyanın kendisidir; `http.client` onu yükleme ilerlemesini bildiren ve
  iptalde hata fırlatan bir sarmalayıcı üzerinden bloklar hâlinde okur. İstek bir
  işçi iş parçacığında çalışır, böylece Deepgram hâlâ düşünürken de İptal çalışır:
  çağıran hemen beklemeyi bırakır, yanıt geldiğinde atılır.
- Her utterance bir segment olur; metni `punctuated_word`'lerden yeniden kurulur,
  böylece kelime sayısı kelime zamanlamalarıyla eşleşir ve satır bölme gerçek
  kelime sınırlarında yapılır. Utterance içermeyen yanıt 0,8 sn'lik
  duraklamalardan bölünür.
- Hatalar: 401/403 → `DeepgramKeyRejectedError`, 402 → `DeepgramCreditError`,
  400 → Deepgram'ın `err_msg`'ını taşıyan `DeepgramRequestError`, 504 →
  `DeepgramTimeoutError` (yeniden denenmez: aynı ses yine zaman aşımına uğrar),
  429/5xx/erişilemez → yeniden denenebilir `DeepgramUnavailableError`.
- Anahtar ([app/core/deepgram_key.py](../app/core/deepgram_key.py)) sırasıyla
  `CAPTIONFORGE_DEEPGRAM_API_KEY` / `DEEPGRAM_API_KEY`, `.env` ve
  `config.json`'ın yanındaki `deepgram.key` dosyasından okunur; dosya `mkstemp`
  ile yazıldığından ilk bayttan itibaren 0600 modundadır. Bir `Config` alanı
  değildir, bu yüzden `config show` ve `persist` onu sızdıramaz; arayüzler
  yalnızca son dört karakterini görür. `check_and_save_key` kaydetmeden önce
  Deepgram'a sorar (`GET /v1/projects`): 401 reddeder, 403 yine gerçek bir
  anahtar sayılır, çevrimdışıyken kontrol edilmeden kaydedilir.

### Whisper adaptörü ([app/adapters/whisper_adapter.py](../app/adapters/whisper_adapter.py))

- `faster_whisper` **tembel** biçimde, `importlib` ile, transkripsiyon anında
  import edilir. Uygulamanın geri kalanı o paket kurulu olmadan çalışır; eksik
  paket `WhisperNotInstalledError` olarak görünür.
- Cihaz: `auto` → `ctranslate2.get_cuda_device_count()` sıfırdan farklıysa
  `cuda`, değilse `cpu`. GPU yokken açıkça `cuda` istenirse sessizce düşmek
  yerine `CudaUnavailableError` fırlatılır.
- Hesaplama tipi: `auto` → CUDA'da `float16`, CPU'da `int8`.
- VAD varsayılan olarak açıktır (`min_silence_duration_ms=500`); ayrıca
  `threshold` ve `speech_pad_ms` yapılandırılabilir (yetersiz dolgu kelime
  başlangıçlarını kırpar).
- **Uzun kayıt çözümleme korumaları** dışa açıldı ve iletiliyor:
  `condition_on_previous_text` burada motorun `true` varsayılanının aksine
  **false**'tur; çünkü çözülen metni pencereler arasında taşımak, saatlerce
  süren seslerde tekrar döngülerinin başlıca nedenidir.
  `compression_ratio_threshold`, `log_prob_threshold` ve `no_speech_threshold`
  motorun bozulma korumalarıdır; `hallucination_silence_threshold` isteğe
  bağlıdır ve kelime zaman damgası gerektirir — ayarlandığında adaptör bunu
  sizin için açar.
- `initial_prompt` (CLI'de `--prompt`) çözücüye beklenen kelime dağarcığını verir.
- **Kelime zaman damgaları** varsayılan olarak açıktır. Her kelimenin başlangıcı,
  bitişi ve olasılığı donmuş bir `WordTiming`'e çevrilip segmentte taşınır;
  son işlemenin bununla ne yaptığı için §8'e bakın.
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
kapalıdır**; siz istemedikçe konuşmacının harflerine dokunulmaz. ≥2 kelimelik
bitişik tekrar eden ifadelerin sadeleştirilmesi
(`_collapse_repeated_phrase`) de `collapse_repeated_phrases` ile **isteğe
bağlıdır** — bir Whisper tekrar döngüsünü, Arap hitabetinin sürekli kullandığı
kasıtlı retorik tekrardan ayırt edemez. Tanınan bir sessizlik cue'su (music/applause/
silence/موسيقى/تصفيق/صمت) yalnızca `no_speech_probability ≥ 0.9` ise atılır.

**2. Tekrar temizliği**
Yalnızca bir önceki segmentle, harf büyüklüğü yok sayılmış temizlenmiş bir
anahtar üzerinden karşılaştırılır. Birebir aynıysa → önceki segmentin bitişi
uzatılır ve yenisi atılır. Aksi hâlde
`SequenceMatcher.ratio() ≥ duplicate_detection_threshold` (0.9) veya tespit
edilen bir baş-ifade örtüşmesi düzeltmeyi tetikler: tekrar eden baş ifade
mevcut metinden kırpılır ya da — mevcut cue tamamen bir öncekinin içinde
kalıyorsa — soğurulup bitiş zamanı uzatılır. YouTube otomatik altyazılarının tipik kayan tekrar
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
yarısı geçildikten sonra cümle sonları tercih edilir.

Yeni zamanlamalar, bir **kelime hizalaması** varsa ve tüm belirteçler birebir
örtüşüyorsa oradan gelir (`_aligned_timings`): her parça ilk kelimesinin
başlangıcını ve son kelimesinin bitişini alır; böylece cue tam olarak kelimeler
söylendiğinde görünür ve gerçek bir duraklama boş kalır. Hizalama yoksa —
indirilen altyazılar ya da temizliğin değiştirdiği metin — `distribute_duration`
devreye girer ve aralığı parça uzunluğuyla orantılı paylaştırır.

Fark büyüktür. Konuşmacının ortasında dört saniye duraksadığı
"بسم الله الرحمن الرحيم" cue'su için:

| | birinci cue | ikinci cue |
|---|---|---|
| kelime hizalı | 0.000 → 1.000 | 5.000 → 6.000 |
| orantısal | 0.000 → 2.667 | 2.667 → 7.000 |

Orantısal bölme, ikinci satırı kimse söylemeden 2,3 sn önce gösteriyor.
Metin temizliği, tekrar ayıklama veya birleştirme belirteç sayısını
değiştirdiğinde hizalamalar tahmin edilmez, **düşürülür**; böylece uyumsuzluk
metni yanlış yerleştirmek yerine eski davranışa geriler.

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
   `--overwrite` yoksa `available_stem`, istenen *her* uzantının boş olduğu ilk
   indekse ilerler; böylece set tek bir gövdede kalır (`title (2).srt` **ve**
   `title (2).vtt`), indeksler karışmaz. Var olan dosyalar asla değiştirilmez ve
   asla hata verdirmez; tamamlanmış bir transkripsiyonu isim çakışmasına kurban
   etmek daha kötü bir sonuçtur. `--overwrite` özgün adı koruyup dosyayı yerinde
   değiştirir. Arama 1000 denemeyle sınırlıdır.
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

**Eskimiş yt-dlp.** YouTube sitesini birkaç haftada bir değiştirir ve eski yt-dlp
sürümlerini reddetmeye başlar: HTTP 403, "Sign in to confirm you're not a bot",
"nsig extraction failed", "Unable to extract" ya da tüm formatların kaybolması
(inceleme sırasında "Requested format is not available"). Yeniden denemek bunu
asla düzeltmez; bu yüzden bu hatalar yeniden denenmez, `ExtractorRefusedError`
olarak işaretlenir (`MetadataRefusedError`, `SubtitleStreamForbiddenError`,
`AudioStreamForbiddenError`, `MediaStreamForbiddenError`).

Adaptör yalnızca reddi bildirir. Ne yapılacağına arayüz karar verir ve iki
arayüz de bir şey kurmadan önce sorar:

- **Terminal.** `_run_with_config` reddi yakalar. `check_for_updates` açıksa,
  pip varsa ve stdin ile stdout bir TTY ise taze bir kontrol yapar ve mevcut
  tüm güncellemeleri listeler. yt-dlp ilk sırada gelir ve varsayılanı evettir;
  diğerlerinin varsayılanı hayırdır. yt-dlp kurulduysa komut bir kez daha
  çalışır. Bir betikten çalışırken soru sormak yerine "Run 'captionforge
  update'" yazdırır.
- **Sayfa.** `/api/inspect` hataları ve başarısız iş anlık görüntüleri
  `update_may_help` taşır. Sayfa bunun üzerine `/api/updates?fresh=1` ister ve
  güncellemeler satırını yt-dlp işaretli olarak gösterir. "Update selected"
  işaretli adları `/api/updates`'e gönderene kadar hiçbir şey kurulmaz.

**Güncellemeler.** `PackageUpdater` ([app/adapters/package_updater.py](../app/adapters/package_updater.py))
sabit bir listeye sahiptir: `PACKAGES`. Bu liste CaptionForge'un doğrudan
bağımlılıklarını içerir. Her birinin `pyproject.toml`'daki sürüm aralığı (bir
test ikisini aynı tutar) ve önerildiği her yerde yanında gösterilen tek satırlık
bir *görevi* vardır.

- `check()`, kurulu olanlar üzerinde `pip install --upgrade --dry-run --report -`
  çalıştırır; neyin daha yeni olduğuna pip'in kendi indeks ayarları ve
  çözümleyicisi karar verir. Hiçbir şey kurulmaz. Taze bir yanıt istenmedikçe
  sonuç altı saat boyunca yeniden kullanılır. Sayfa açılışta sorar; bir ret ve
  `captionforge update` taze sorar.
- `install(names)` yalnızca `PACKAGES` içindeki adları kabul eder, bir isteğin
  gönderdiklerini asla kabul etmez, ve tam olarak onları yükseltir.
- Yeni bir yt-dlp yerinde yüklenir: tüm `yt_dlp` modülleri `sys.modules`'tan
  çıkarılıp yeniden içe aktarılır, böylece açık bir sayfa onu hemen kullanır.
  Adaptörün `yt_dlp.YoutubeDL` ve `DownloadError`'ı her çağrıda yeniden araması
  bu yüzdendir. Diğer paketler zaten içe aktarılmışlarsa `restart_needed`
  bildirir.
- Süreç başına tek bir güncelleyici (`UPDATER`) ve tek bir kilit vardır; bir
  kontrol ile bir kurulum asla aynı anda pip çalıştırmaz.
- pip yoksa (veya paketlenmiş bir çalıştırılabilir dosyada) hiçbir şey
  çalışmaz; `doctor` bunu söyler.

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
`WhisperAdapter(model_factory=…, cuda_detector=…)`, `DeepgramAdapter(opener=…)`
ve `retry_call(sleep=…)`. Bir Deepgram testi, yüklemenin eksiksiz aktığını
kanıtlamak için `127.0.0.1` üzerinde gerçek bir HTTP sunucusu çalıştırır;
hiçbiri Deepgram'a ulaşmaz.

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

Kasıtlı sınırlar: yalnızca tekil, canlı olmayan videolar veya tek tek dosyalar;
oynatma listesi, kanal, klasör, canlı yayın, çeviri, konuşmacı ayrıştırma veya
çerez/kimlikli erişim yok; agresif yazım/dilbilgisi yeniden yazımı yok. Bir
dosyaya gömülü altyazı izleri (örneğin bir MKV'ninkiler) okunmaz: dosya her
zaman transkribe edilir.

Mevcut koddaki, bilinmesinde fayda olan pürüzler:

- `prepare-audio` hâlâ "Transcription will be implemented in Phase 5" yazıyor.
  Bayat bir mesaj — `transcribe` bunu bugün zaten yapıyor.
- yt-dlp, `quiet: True` olmasına rağmen altyazı indirme sırasında kendi
  `[download]` ilerleme ve `ERROR:` satırlarını doğrudan terminale yazıyor. Bu,
  "ham yt-dlp metni asla stdout'a ulaşmaz" güvencesiyle çelişiyor; `noprogress`
  ayarlanmamış.
- `configure_logging`'in docstring'i konsol ve dosya handler'ından söz ediyor,
  ancak yalnızca dosya handler'ı kaydediliyor.
- `clean` komutu `ExportService`'i yeniden kullanabilmek için girdi dosyası için
  bir `VideoMetadata` kuruyor; sonuç sonrasında istenen hedefe yeniden
  adlandırılıyor.
