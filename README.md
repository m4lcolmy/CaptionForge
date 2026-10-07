# CaptionForge

CaptionForge downloads and exports existing YouTube captions. When no matching
caption exists, it prepares mono 16 kHz audio and transcribes it locally with
`faster-whisper`, or, if you pick it, has [Deepgram](https://deepgram.com)
transcribe it online with your own API key. It transcribes video and audio files
already on your computer the same way. It can also save a YouTube video itself
as an MP4 at a chosen quality, or its audio alone as an MP3.

The caption and transcription workflows still never fetch the video stream:
whole-file downloads happen only when you ask for one, through `download` or
the page's download row.

Transcription runs on this computer unless you pick Deepgram. Deepgram is the
only path that sends anything of yours off it: a compressed copy of the audio,
and only when a caption track is not being reused. Whisper stays the default.

## Features

- Run the whole workflow from a browser on your own machine
- Transcribe a video or audio file from this computer: choose it, drop it, or
  paste its path
- Or install it as a desktop app and start it from your applications list
- Inspect video metadata and available caption tracks
- Select captions by preferred language
- Save the video as MP4 at any quality it publishes, or the audio as MP3
- See each quality's approximate size before starting the download
- Export SRT, VTT, TXT, JSON, or DOCX
- Generate multiple formats in one command
- Create plain or timestamped TXT transcripts
- Save a Word transcript as plain prose, with right-to-left support
- Preserve Arabic words, punctuation, and diacritics
- Produce safe filenames and never replace existing output
- Prepare audio-only fallback input with yt-dlp and FFmpeg
- Clean per-job intermediate files, or preserve them on request
- Automatically select CPU or NVIDIA CUDA and a suitable compute type
- Transcribe locally with VAD, timestamps, progress, and cancellation support
- Or transcribe online with Deepgram Nova-3, sending compressed audio only
- Conservatively clean Arabic, Latin, and mixed-language subtitle text
- Repair timing, remove repetition, and format subtitles to two readable lines
- Retry temporary network failures without retrying invalid input
- Persist validated settings and write output files atomically
- Keep rotating technical logs separate from concise CLI errors

CaptionForge works with individual, non-live YouTube videos, and with single
video or audio files on this computer (anything FFmpeg can read: MP4, MKV, MOV,
WebM, MP3, M4A, WAV, FLAC, OGG, Opus, and more).

## Installation

Python 3.12 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pip install -e ".[transcription]"
python -m pip install -e ".[web]"
```

The desktop app is a native Qt window. It needs only its own extra, not the
web packages:

```bash
python -m pip install -e ".[desktop]"
```

## Web interface

Serve the interface to a browser on this computer:

```bash
captionforge web
```

CaptionForge prints a `http://127.0.0.1:PORT/?t=TOKEN` link and opens it. Paste a
video link, pick a language and formats, and download the results. Nothing is
uploaded anywhere unless you pick Deepgram: yt-dlp, FFmpeg, and faster-whisper
all run locally, exactly as they do for the commands below.

The choices people change most sit in the card itself, under **Transcribe
with**: Whisper (this computer) or Deepgram (online), a dropdown of that
engine's models, and **Transcribe even when the video has captions**. The
sentences around the button follow them, so the button reads **Transcribe**
once the override is ticked, and the track note says what will happen instead.
Everything rarer (machine-translated tracks, TXT timestamps, replacing files,
cleanup, keeping audio, the device, names and spellings) is under **More
options**.

A file on this computer works too. Click **Choose file**, drop the file anywhere
on the page, or paste its path into the link field. A file has no caption tracks
to reuse, so the page skips that section, and its button reads **Transcribe**.
The transcript is named after the file: `lecture.mp4` gives `lecture.srt`.

A browser never tells a page where a chosen or dropped file lives, so the page
hands CaptionForge a copy, kept in `temp/uploads/` on this computer. The copy is
removed when you choose another file, and when CaptionForge stops. Pasting the
path instead reads the file where it is, with no copy, which is the better
choice for a large video.

Looking a video up also fills in a **Download the file** row: one chip per
quality that video actually publishes, each showing its approximate size, with
MP3 first. One click starts that download; the same progress bar, Cancel button,
and results list serve it. A download and a transcription run on separate
workers, so asking for an MP3 never waits behind an hour of Whisper.

```bash
captionforge web --port 8800    # bind a fixed port instead of a free one
captionforge web --no-open      # print the link without opening a browser
```

The page remembers the language, formats, engine, each engine's model, device,
options, and vocabulary hint you last used, and opens with them next time. They are kept in
`$XDG_CONFIG_HOME/captionforge/web-preferences.json`, beside `config.json` but
separate from it: interface choices never change what the CLI does. Delete that
file to start from the configured defaults again. Browser storage is not used,
because `captionforge web` binds a new port each run and that changes the page's
origin.

The server binds `127.0.0.1` only, never `0.0.0.0`. Every `/api` call must carry
the per-process token from that link, requests addressed to any other hostname
are refused, and downloads are restricted to the files the job actually wrote.
The page loads no fonts, scripts, or styles from the internet, so it works with
the network unplugged once a caption track or model is already local.

## Desktop app

Add CaptionForge to this computer's applications, so it starts from an icon
instead of a terminal:

```bash
captionforge install-desktop
```

Search your applications for **CaptionForge** and start it like any other app.
It opens a native window, drawn with Qt, that looks and works like the page
`captionforge web` serves: the same sections, the same wording, the same green
ramp, light or dark with your system. There is no server behind it, no port,
and no browser; the window calls CaptionForge directly.

The entry runs `captionforge desktop` from the folder you were in when you
installed it, so output, `temp/`, and `logs/` land exactly where they land when
you run the commands by hand. Install from somewhere else with `--workdir`, and
undo the whole thing with `--remove`:

```bash
captionforge install-desktop --workdir ~/Videos/CaptionForge
captionforge install-desktop --remove
```

You can also start the app without installing anything:

```bash
captionforge desktop
```

A few things differ from the page, because an application can do better:

- **Finished files open with one click.** They are already in the output
  folder, so a file's row opens it with this computer's usual app, and the
  folder named under the list opens the folder. Nothing is downloaded twice.
- **A file is read where it lies.** Choose file opens this computer's own file
  chooser, and a file dropped anywhere on the window is taken. Either way, the
  window reads it in place instead of copying it first.
- **Starting it twice does not start two of them.** The second launch brings
  the open window forward and exits.
- **Closing the window never abandons work.** A download or a transcription that
  is already running finishes and writes its files first. The window hides
  meanwhile, and comes back if you start CaptionForge again.

On Linux this writes a `.desktop` entry and an icon under
`~/.local/share`; on macOS a small bundle in `~/Applications`; on Windows a
Start Menu shortcut. Only the Linux entry has been verified on real hardware.

Coming from 0.7, where the window was a Chrome app: run
`captionforge install-desktop` once more, so the entry matches the new window
and the dock shows its icon, and delete
`~/.local/share/captionforge/browser-window`, the browser profile the old
window used.

## Deepgram

Deepgram is an online speech-to-text service. CaptionForge can use it instead of
Whisper when you want its accuracy or speed, or when this computer is slow. It
costs Deepgram credit, and it needs an API key from your Deepgram account
(Console → API Keys; the Member role or higher).

Save the key once, in any of these ways:

- On the page or in the window: pick **Deepgram**, paste the key, and click
  **Save**.
- In a terminal: `captionforge deepgram-key`, which asks for it without
  showing what you type. `captionforge deepgram-key --forget` removes it.
- In the environment: `CAPTIONFORGE_DEEPGRAM_API_KEY` or `DEEPGRAM_API_KEY`,
  also read from `.env`. These take priority over a saved key.

A pasted key is checked with Deepgram before it is saved, and a key Deepgram
refuses is never saved. Offline, it is saved and checked again on first use. The
saved key lives in `~/.config/captionforge/deepgram.key`, readable only by you.
It is deliberately not a `config` setting, so `config show` never prints it. The
page and the window only ever show its last four characters.

What is sent, and when:

- Only when a transcription actually runs. A video whose caption track is
  reused sends nothing, and needs no key.
- Only the audio, as mono Opus at 48 kbps: about 21 MB per hour. The page and
  the window show the size for the video you looked up.
- The language you set is sent as such. Leave the field empty and Deepgram
  detects it, from about 35 languages.
- **Names and spellings** go as Deepgram key terms, one per comma.

Nova-3 is the default model and covers Arabic, its regional variants, Turkish,
and many more. Any other Deepgram model can be named under **Something else…**
or with `--model`.

```bash
captionforge transcribe ~/Videos/lecture.mp4 --engine deepgram --language tr
captionforge config set transcription_engine deepgram   # make it the default
```

## Usage

Inspect a video and its caption tracks:

```bash
captionforge inspect "https://youtu.be/qJFbKl6RjLU?si=wdoe8oQzasIgydBk" --language ar
```

Download captions using the configured default formats:

```bash
captionforge extract "https://youtu.be/qJFbKl6RjLU?si=wdoe8oQzasIgydBk" --language ar
```

Choose one or more formats and an output directory:

```bash
captionforge extract "https://youtu.be/qJFbKl6RjLU?si=wdoe8oQzasIgydBk" \
  --format srt \
  --format txt \
  --output ./output
```

Save a Word document instead of a subtitle file:

```bash
captionforge extract "https://youtu.be/qJFbKl6RjLU?si=wdoe8oQzasIgydBk" \
  --language ar --format docx --output ./output
```

The `docx` format writes the transcript as plain readable prose: no cue numbers
and no timestamps. Consecutive cues are joined into paragraphs that break at a
silent gap, at a sentence boundary, or at a length limit. The video title becomes
the document heading, and Arabic (or any other right-to-left language) is written
right-aligned with the correct text direction. `--timestamped-txt` affects TXT
only and never adds timings to the Word document.

Export captions when available, otherwise transcribe locally (or with Deepgram,
using `--engine deepgram`):

```bash
captionforge transcribe "https://youtu.be/qJFbKl6RjLU" \
  --language ar --model small --device auto --compute-type auto \
  --format srt --format txt --output ./output
```

Transcribe a video or audio file on this computer instead of a link:

```bash
captionforge transcribe ~/Videos/lecture.mp4 --language ar --format srt
captionforge transcribe "/home/me/Voice notes/meeting.m4a" --format docx
```

A file has no captions to reuse, so it is always transcribed. The file itself
is read, never moved or changed; the transcript lands in the output folder,
named after the file. `inspect` shows a file's length, and a file with no sound
in it is refused before any model loads. `extract` and `download` are for
YouTube only, and say so if given a file.

The command reuses a suitable YouTube caption by default. Use `--force` to
transcribe anyway, `--keep-audio` to preserve prepared audio, and `--overwrite` to
replace output files.

Existing output is never replaced or refused. When the natural filename is
taken, the whole set of generated files moves to the next free variant —
`title (2).srt`, `title (3).srt` — so a long transcription is never discarded
over a name clash. Pass `--overwrite` to replace the originals in place
instead. The default model is `small`, avoiding an impractical
large-model default on low-resource machines.

Transcription quality options:

- `--prompt` supplies a vocabulary hint. Proper nouns and domain terms in the
  prompt measurably improve how Whisper spells them. Deepgram receives them as
  key terms, one per comma.
- `whisper_word_timestamps` (default on) makes CaptionForge cut subtitle lines
  on real word boundaries instead of estimating them from character counts. It
  costs a little speed and is what `whisper_hallucination_silence_threshold`
  needs in order to work.
- `whisper_condition_on_previous_text` defaults to **false**, unlike the
  faster-whisper default. Carrying decoded text between windows is the usual
  cause of repetition loops and drift on long recordings; the cost is slightly
  less cross-window context. Set it to `true` to restore the engine default.
- `whisper_compression_ratio_threshold`, `whisper_log_prob_threshold` and
  `whisper_no_speech_threshold` are the engine's degeneracy guards, now tunable.

Save the video itself, or just its sound:

```bash
captionforge download "https://youtu.be/VIDEO_ID" --list          # what it offers
captionforge download "https://youtu.be/VIDEO_ID"                 # best MP4
captionforge download "https://youtu.be/VIDEO_ID" --quality 720   # 720p MP4
captionforge download "https://youtu.be/VIDEO_ID" --quality mp3   # MP3 audio
```

`--quality` takes `mp3`, `best`, or a height. A height that a video does not
publish steps down to the best one below it rather than failing, so `--quality
1080` still works on a video that stops at 720p. MP4 downloads prefer H.264
video and AAC audio: YouTube also offers VP9 inside an MP4 and yt-dlp rates it
higher, but a `.mp4` that QuickTime and ordinary video editors refuse to open is
not what an MP4 download should hand back.

Files are named after the video, with the quality in brackets for video
(`title [720p].mp4`) and without one for audio (`title.mp3`), since there is
only one audio quality. As everywhere else, an existing file is numbered rather
than replaced unless `--overwrite` is passed. Sizes shown before a download are
estimates read from the stream metadata, usually within a few percent.

`captionforge prepare-audio` remains available for audio-only diagnostics.

CaptionForge never updates anything without asking. `captionforge update` lists
newer releases of the packages it uses, each with a line saying what that
package does, and asks about each one before installing it:

```bash
captionforge update        # choose one by one
captionforge update --yes  # install everything listed
```

The page asks the same question: when it opens and something newer exists, an
"Updates available" row lists the packages with checkboxes, all unticked.
Updates stay inside the version ranges in `pyproject.toml`. A new yt-dlp is
used straight away; other packages take effect the next time CaptionForge
starts. `captionforge config set check_for_updates false` stops CaptionForge
from looking by itself; `captionforge update` still works.

Run `captionforge doctor` to check FFmpeg and FFprobe, yt-dlp, faster-whisper,
python-docx, CUDA, the detected GPU, recommendations, folder access, the
transcription engine, and whether a Deepgram key is set. Doctor never downloads a
model and never contacts Deepgram.

Post-processing is enabled for both downloaded captions and Whisper results. Use
`--no-postprocess` with `extract` or `transcribe` when source segmentation must be
retained. Post-processing removes redundant cues but never rewrites speech:
collapsing an adjacent repeated phrase is opt-in via `collapse_repeated_phrases`,
because deliberate rhetorical repetition is indistinguishable from an artifact. Clean an existing file without downloading or transcribing:

```bash
captionforge clean captions.srt
captionforge clean captions.vtt --output ./output
```

The clean command preserves the input format and writes `*.cleaned.srt` or
`*.cleaned.vtt` unless another destination is supplied; it does not convert to
DOCX.

## Configuration

Settings are persisted in
`$XDG_CONFIG_HOME/captionforge/config.json` (normally
`~/.config/captionforge/config.json`). Environment variables override persisted
values.

```bash
captionforge config show
captionforge config set retry_count 4
captionforge config set retry_delay_seconds 2
captionforge config reset
```

```text
CAPTIONFORGE_AUDIO_FORMAT=wav
CAPTIONFORGE_AUDIO_SAMPLE_RATE=16000
CAPTIONFORGE_AUDIO_CHANNELS=1
CAPTIONFORGE_TEMP_DIRECTORY=temp
CAPTIONFORGE_KEEP_TEMP_FILES=false
CAPTIONFORGE_FFMPEG_EXECUTABLE=ffmpeg
CAPTIONFORGE_DEFAULT_WHISPER_MODEL=small
CAPTIONFORGE_WHISPER_DEVICE=auto
CAPTIONFORGE_WHISPER_COMPUTE_TYPE=auto
CAPTIONFORGE_WHISPER_BEAM_SIZE=5
CAPTIONFORGE_WHISPER_VAD_ENABLED=true
CAPTIONFORGE_WHISPER_MIN_SILENCE_DURATION_MS=500
CAPTIONFORGE_WHISPER_VAD_THRESHOLD=0.5
CAPTIONFORGE_WHISPER_VAD_SPEECH_PAD_MS=400
CAPTIONFORGE_WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
CAPTIONFORGE_WHISPER_INITIAL_PROMPT=
CAPTIONFORGE_WHISPER_WORD_TIMESTAMPS=true
CAPTIONFORGE_WHISPER_COMPRESSION_RATIO_THRESHOLD=2.4
CAPTIONFORGE_WHISPER_LOG_PROB_THRESHOLD=-1.0
CAPTIONFORGE_WHISPER_NO_SPEECH_THRESHOLD=0.6
CAPTIONFORGE_WHISPER_HALLUCINATION_SILENCE_THRESHOLD=
CAPTIONFORGE_WHISPER_LANGUAGE=
CAPTIONFORGE_WHISPER_MODEL_DOWNLOAD_DIRECTORY=
CAPTIONFORGE_TRANSCRIPTION_ENGINE=whisper
CAPTIONFORGE_DEEPGRAM_MODEL=nova-3
CAPTIONFORGE_MAXIMUM_CHARACTERS_PER_LINE=42
CAPTIONFORGE_MAXIMUM_SUBTITLE_LINES=2
CAPTIONFORGE_MINIMUM_SUBTITLE_DURATION=0.8
CAPTIONFORGE_MAXIMUM_SUBTITLE_DURATION=7.0
CAPTIONFORGE_SUBTITLE_MERGE_THRESHOLD=1.0
CAPTIONFORGE_DUPLICATE_DETECTION_THRESHOLD=0.9
CAPTIONFORGE_REMOVE_DIACRITICS=false
CAPTIONFORGE_COLLAPSE_REPEATED_PHRASES=false
CAPTIONFORGE_NORMALIZE_ARABIC_LETTERS=false
CAPTIONFORGE_NORMALIZE_ARABIC_INDIC_DIGITS=false
CAPTIONFORGE_RETRY_COUNT=3
CAPTIONFORGE_RETRY_DELAY_SECONDS=1.0
CAPTIONFORGE_MINIMUM_FREE_DISK_BYTES=104857600
CAPTIONFORGE_CHECK_FOR_UPDATES=true
```

`CAPTIONFORGE_CONFIG_FILE` may point to another config file. The Deepgram key is
not one of these settings; see [Deepgram](#deepgram). Invalid persisted
values fall back individually to safe defaults; invalid values passed to
`config set` are rejected.

## Errors, retries, and logs

Normal CLI output contains a short actionable error, never raw yt-dlp, FFmpeg,
Whisper, CUDA, or Python details. Technical causes, job identifiers, stages,
selected methods, model/device choices, retries, output paths, and durations are
written to `logs/`. Logs rotate at 10 MB and are retained for 14 days.

CaptionForge retries translated temporary metadata, caption, audio-download,
model-load, and Deepgram connection failures. Invalid URLs, unsupported resources, unavailable videos,
bad timestamps, missing FFmpeg, invalid model names, and invalid output paths
are not retried. Interrupting a job cancels it and removes temporary and partial
files unless preservation was requested.

### Troubleshooting

- Run `captionforge doctor` to check FFmpeg, CUDA, dependencies, disk paths, and
  write access.
- **"FFprobe … is not installed"** when looking a file up: FFprobe ships with
  FFmpeg, and CaptionForge uses it to read a file's length and check it has
  sound. Install the full FFmpeg package. With `ffmpeg_executable` set to a
  custom path, FFprobe is expected in the same folder.
- **HTTP 403 on audio or captions** means yt-dlp can no longer read YouTube's
  current site. Retrying never fixes it; a newer yt-dlp does. When this
  happens in a terminal, CaptionForge checks for one, shows it, and asks
  "Update yt-dlp?". If you say yes, it installs it and runs your command once
  more. On the page, the same refusal brings up the updates row with yt-dlp
  ticked, waiting for "Update selected". If yt-dlp is already the newest
  release, YouTube is probably limiting your connection; try again later. From
  a script (no terminal to ask), run `captionforge update` afterwards.

  yt-dlp also wants a JavaScript runtime (`deno`) for signature extraction and
  has deprecated working without one. `doctor` reports whether you have it.
- **"a required NVIDIA runtime library is missing"** means a GPU was detected
  but CUDA's math libraries could not be loaded. Install them:

  ```bash
  python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
  ```

  These wheels place their shared objects in `site-packages/nvidia/*/lib`, a
  directory the dynamic loader never searches, so being installed is not enough
  on its own. CaptionForge loads them from there by absolute path at startup,
  which is why no `LD_LIBRARY_PATH` is needed. If `doctor` still reports them as
  missing, the wheels are genuinely absent or the GPU is unusable; run with
  `--device cpu` in the meantime. `doctor` names the specific libraries it could
  not load.
- **Deepgram errors** say what to do. A refused key (401 or 403) needs a new key
  with the Member role or higher, and is never retried. "Out of credit" (402)
  needs a top-up in the Deepgram console. A refused request names Deepgram's
  reason, usually a language the chosen model does not cover. Deepgram stops
  working on a request after 10 minutes, so an extremely long recording may be
  refused as too long; Whisper has no such limit.
- Increase `retry_count` or `retry_delay_seconds` for throttling and unstable
  connections.
- For GPU memory errors, use a smaller model, `--compute-type int8`, or
  `--device cpu`.
- Use a writable `--output` directory and ensure both output and temporary
  filesystems have enough free space.
- Inspect the newest file in `logs/` when the CLI asks for technical details.

Useful options:

```text
--language ar       Preferred caption language
--format FORMAT     srt, vtt, txt, json, or docx; may be repeated
--output PATH       Output directory
--timestamped-txt   Add timestamps to TXT output
--overwrite         Replace existing output files instead of numbering
--no-postprocess    Bypass Phase 6 processing
--allow-translated  Also consider machine-translated caption tracks
--engine ENGINE     whisper (this computer) or deepgram (online)
--model NAME        A Whisper model or folder, or a Deepgram model
--prompt TEXT       Names and terms: a hint for Whisper, key terms for Deepgram
```

Run `captionforge --help` or `captionforge extract --help` for the complete
command reference.

## Caption selection

CaptionForge prefers tracks in this order:

1. Exact manual language match
2. Manual base-language match
3. Exact automatic language match
4. Automatic base-language match

For example, requesting `ar-EG` can match another Arabic variant when an exact
track is unavailable.

YouTube also publishes machine translations of its automatic transcription for
roughly 150 languages. These are translations of a transcription and rank
*below* having no track at all, so `transcribe` falls back to transcription
instead of exporting them. `inspect` marks them in a `Translated` column, and
`--allow-translated` opts back in.

Caption tracks are downloaded as `json3` where available. YouTube's automatic
VTT is a rolling format in which each cue repeats the previous line, roughly
tripling the word count and losing the true start of the first phrase; `json3`
carries one cue per phrase and needs no repair.

## Development

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pytest --cov=app --cov-report=term-missing
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check app tests
.venv/bin/python -m mypy app
```

The default test suite is offline. The optional live YouTube integration test
requires `CAPTIONFORGE_INTEGRATION_VIDEO_URL`. The optional real Whisper smoke
test requires `CAPTIONFORGE_INTEGRATION_AUDIO`; it is excluded by default.

Run optional integrations explicitly:

```bash
.venv/bin/python -m pytest -m integration
```

## Limitations

- No live streams, playlists, folders of files, or translation
- Subtitle tracks embedded in a file (an MKV's, for example) are not read; a
  file is always transcribed
- MP4 downloads offer 360p to 2160p, and only the heights a video publishes; other heights are not re-encoded into existence
- The web interface covers `extract`, `transcribe` (for links and files), and `download` only; settings, `clean`, and `doctor` stay on the command line
- The desktop app offers what the page offers; its applications-menu entry has been verified on Linux only
- No authenticated or cookie-based access
- No speaker diarization, translation, or aggressive spelling/grammar rewriting
- Cancelling while Deepgram is transcribing stops CaptionForge waiting at once,
  but Deepgram finishes, and bills, a request whose audio it already has

For implementation details, see the
[Phase 6 report](docs/phase-6-report.md),
[Phase 7 report](docs/phase-7-report.md),
[Phase 5 report](docs/phase-5-report.md),
[Phase 4 report](docs/phase-4-report.md),
[Phase 3 report](docs/phase-3-report.md), and
[Phase 2 report](docs/phase-2-report.md).
