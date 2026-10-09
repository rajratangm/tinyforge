# Linux, Windows और Colab पर tinyforge इंस्टॉल करें और उपयोग करें

कमांड हर जगह एक जैसे हैं; केवल शेल की सिंटैक्स अलग है। हर चरण में **Linux / macOS (bash)** और
**Windows (PowerShell)** दोनों साथ-साथ दिखाए गए हैं।

## कहाँ क्या टेस्ट किया गया है

| प्लेटफ़ॉर्म | स्थिति |
|---|---|
| **Linux + NVIDIA GPU** (Google Colab, Ubuntu, Tesla T4 16 GB, Python 3.13) | प्रकाशित पैकेज से शुरू से अंत तक चलाया गया: इंस्टॉल, `doctor`, `memory`, डेटा टूल, शुरू से पाइपलाइन, LoRA फ़ाइन-ट्यून, job-spec वर्कर, सर्वर (OpenAI API, सीक्रेट गार्डरेल, मेट्रिक्स), GGUF एक्सपोर्ट और llama.cpp सर्विंग। देखें `notebooks/colab_smoke_test.ipynb`। |
| **Linux, केवल CPU** (Docker और GitHub Actions, Python 3.10 और 3.12) | पूरा टेस्ट सूट CI में पास होता है। |
| **Windows 11 + NVIDIA GPU** (RTX 3050 Ti 4 GB, Python 3.12) | मुख्य डेवलपमेंट मशीन; README के "Verified results" में दिया सब कुछ यहीं चलाया गया था। |
| **Linux, 2 NVIDIA GPU** (Kaggle 2x T4) | डेटा-पैरेलल LoRA सत्यापित: NCCL काम करता है, रेप्लिका एक जैसे रहते हैं, रीज़्यूम काम करता है; छोटे जॉब पर थ्रूपुट एक GPU का केवल 1.2x-1.4x रहा। |
| मल्टी-नोड, GPU वाला Kubernetes, AMD (ROCm), Apple silicon | **टेस्ट नहीं किया गया।** Kubernetes चार्ट और ऑपरेटर के हिस्से केवल CPU-वाले `kind` क्लस्टर पर सत्यापित किए गए थे। |

## 1. पूर्व-आवश्यकताएँ

- Python 3.10 या नया (3.10, 3.12 और 3.13 पर टेस्ट किया गया)।
- GPU उपयोग के लिए: चालू ड्राइवर वाला NVIDIA GPU (`nvidia-smi` काम करना चाहिए)। आपको CUDA टूलकिट की ज़रूरत **नहीं** है;
  PyTorch के wheel अपना CUDA रनटाइम साथ लाते हैं।
- Windows: python.org से Python इंस्टॉल करें ("Add to PATH" पर टिक करें)। Linux: `sudo apt install python3 python3-venv`
  (Debian/Ubuntu) या आपकी डिस्ट्रीब्यूशन का समकक्ष कमांड।

## 2. एनवायरनमेंट बनाएँ और इंस्टॉल करें

**Linux / macOS (bash)**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU. For CPU only: .../whl/cpu
pip install "tinyforge[finetune]"
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1      # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # GPU. For CPU only: .../whl/cpu
pip install "tinyforge[finetune]"
```

एक्स्ट्रा (Extras): `[finetune]` LoRA/QLoRA, `[docs]` PDF/Word/Excel इन्जेशन, `[bench]` मानक बेंचमार्क,
`[all]` सब कुछ। इसके बजाय सोर्स से इंस्टॉल करने के लिए: क्लोन के भीतर से `pip install -e ".[dev]"`।

नतीजा जाँचें:

```
tinyforge --version
tinyforge doctor        # hardware, what is installed, and the exact command for anything missing
```

Google Colab और Kaggle पर torch पहले से इंस्टॉल होता है: torch वाली लाइन छोड़ दें। (उनका पहले से इंस्टॉल `torchao`
`peft` के लिए बहुत पुराना है; tinyforge उसे अपनी प्रोसेस से अपने-आप छिपा देता है। इसे बंद करने के लिए `TINYFORGE_KEEP_TORCHAO=1` सेट करें।)

## 3. पहला रन

ये दोनों सिस्टम पर एक जैसे काम करते हैं।

```
tinyforge pipeline --preset micro --steps 300       # from scratch: data -> train -> eval gates (about 1-3 min on a GPU)
tinyforge generate "ROMEO:" --int8
tinyforge ft pipeline --steps 100                    # fine-tune SmolLM2-360M with LoRA (downloads about 700 MB)
tinyforge ft generate "Explain hash tables" --base   # the original model; drop --base for the tuned one
```

आउटपुट मौजूदा डायरेक्टरी के अंदर `runs/` और `data/` में जाते हैं (`runs/micro`, `runs/ft`, ...)। Windows पर वही
पाथ `runs\micro\best.pt` होते हैं; आर्गुमेंट में दोनों स्लैश शैलियाँ चलती हैं।

## 4. मॉडल सर्व करें और उसे कॉल करें

सर्वर localhost से बंधा होता है। ऑथेंटिकेशन डिफ़ॉल्ट रूप से चालू है, इसलिए या तो टोकन सेट करें या, केवल
लोकल प्रयोगों के लिए, इसे बंद कर दें।

**Linux / macOS (bash)**

```bash
export TINYFORGE_API_TOKEN="$(openssl rand -hex 24)"
tinyforge serve --port 8000                     # UI at http://127.0.0.1:8000
curl -s http://127.0.0.1:8000/healthz
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $TINYFORGE_API_TOKEN" -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'
# local experiments only, no token:   TINYFORGE_AUTH=off tinyforge serve
```

**Windows (PowerShell)**

```powershell
$env:TINYFORGE_API_TOKEN = -join ((48..57 + 97..102) | Get-Random -Count 48 | ForEach-Object {[char]$_})
tinyforge serve --port 8000                     # UI at http://127.0.0.1:8000
curl.exe -s http://127.0.0.1:8000/healthz
curl.exe -s http://127.0.0.1:8000/v1/chat/completions `
  -H "Authorization: Bearer $env:TINYFORGE_API_TOKEN" -H "Content-Type: application/json" `
  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":16}'
# local experiments only, no token:   $env:TINYFORGE_AUTH = "off"; tinyforge serve
```

`/v1/chat/completions` एंडपॉइंट OpenAI-संगत है, इसलिए मौजूदा OpenAI क्लाइंट लाइब्रेरी `base_url` को
`http://127.0.0.1:8000/v1` पर इंगित करके काम करती हैं। जिस प्रॉम्प्ट में क्रेडेंशियल जैसी स्ट्रिंग हो, उसे 400 के साथ अस्वीकार कर दिया जाता है।

localhost के अलावा किसी और पते पर सुनने के लिए आपको TLS (`--ssl-certfile`, `--ssl-keyfile`) और एक टोकन चाहिए; देखें
`docs/networking.md`।

## 5. वैकल्पिक हिस्से

| आप क्या चाहते हैं | Linux / macOS | Windows |
|---|---|---|
| दस्तावेज़ (PDF, Word, Excel) ट्रेनिंग डेटा के रूप में | `pip install "tinyforge[docs]"` | वही |
| मानक बेंचमार्क (`bench run`) | `pip install "tinyforge[bench]"` | वही |
| 4-बिट (QLoRA) | `[finetune]` में शामिल (bitsandbytes) | शामिल; नया bitsandbytes wheel चाहिए |
| फ़्यूज़्ड Triton कर्नेल | `pip install triton` (आमतौर पर torch के साथ इंस्टॉल होता है) | `pip install triton-windows` |
| GGUF एक्सपोर्ट और llama.cpp इंजन | llama.cpp बनाएँ या डाउनलोड करें, फिर `export TINYFORGE_LLAMACPP_SRC=~/llama.cpp TINYFORGE_LLAMACPP_BIN=~/llama.cpp/build/bin TINYFORGE_LLAMA_SERVER=~/llama.cpp/build/bin/llama-server` | llama.cpp का रिलीज़ डाउनलोड करें, फिर `$env:TINYFORGE_LLAMACPP_SRC="C:\llama.cpp"` आदि (वही तीन वेरिएबल) |
| VRAM से बड़े मॉडल ट्रेन करना (`backend: soup`) | `soup-cli` को उसके **अपने** venv में इंस्टॉल करें, फिर `export TINYFORGE_SOUP_BIN=/path/to/soup` | वही, `$env:TINYFORGE_SOUP_BIN="C:\path\soup.exe"` |

Linux पर llama.cpp बनाना (एक्सपोर्ट और क्वांटाइज़ेशन के लिए CPU काफ़ी है; GPU सर्विंग के लिए `-DGGML_CUDA=ON` जोड़ें):

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
cmake -S ~/llama.cpp -B ~/llama.cpp/build -DLLAMA_CURL=OFF && cmake --build ~/llama.cpp/build -j --target llama-quantize llama-server
pip install -r ~/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt   # use a separate venv: it pins protobuf
```

फिर: `tinyforge export gguf --base <local HF model folder> --adapter runs/ft/best --quant q8_0` और
`tinyforge serve --engine llamacpp --gguf out/gguf/base-q8_0.gguf --lora-gguf out/gguf/adapter-f16.gguf`।

## 6. स्पेक से ट्रेनिंग जॉब चलाएँ (जो Kubernetes एजेंट चलाता है)

```
tinyforge worker run --spec spec/examples/sql-finetune.yaml --out jobs/sql-1 --dry-run   # validate, change nothing
tinyforge worker run --spec spec/examples/sql-finetune.yaml --out jobs/sql-1
```

एग्ज़िट कोड `spec/worker-contract.md` में सूचीबद्ध हैं (उदाहरण के लिए 75 का अर्थ है "मुझे दोबारा चलाओ")।

## 7. Linux और Windows के बीच जानने योग्य अंतर

- **GPU प्रिसिज़न।** Ampere से पुराने GPU (कंप्यूट क्षमता 8.0 से कम, जैसे T4) में नेटिव bf16 नहीं होता;
  tinyforge उन्हें लॉस स्केलिंग (loss scaling) के साथ fp16 में ट्रेन करता है। Ampere और नए GPU bf16 का उपयोग करते हैं।
- **मल्टी-GPU।** `tinyforge ft train --gpus 2` (या जॉब स्पेक में `resources: {gpus: 2}`) एक ही मशीन पर
  डेटा-पैरेलल तरीके से LoRA ट्रेन करता है: हर GPU अपने पूरे बैच पर ट्रेन करता है (इसलिए प्रभावी बैच N गुना बड़ा होता है; `--split-batch` एक ही बैच को बाँटता है) और छोटे LoRA ग्रेडिएंट हर स्टेप पर औसत किए जाते हैं। इसके लिए
  Linux चाहिए (Windows पर NCCL नहीं है)। CPU पर असली 2-प्रोसेस रन और Kaggle 2x T4 मशीन पर टेस्ट किया गया
  (`notebooks/kaggle_multi_gpu.ipynb`): GPU एक जैसे रहते हैं और रीज़्यूम काम करता है, लेकिन छोटे 360M जॉब पर थ्रूपुट एक
  GPU का केवल 1.2x-1.4x रहा, 2x नहीं। बड़े मॉडल, जहाँ कंप्यूट हावी रहता है, बेहतर स्केल होने चाहिए; यह टेस्ट नहीं किया गया है। उपलब्ध नहीं: एक मॉडल को कई GPU में शार्ड करना (FSDP),
  मल्टी-नोड, और एक से अधिक GPU के साथ `soup` बैकएंड।
- **Triton, JAX, vLLM, DeepSpeed।** पहले Linux। Windows पर `triton-windows` चाहिए; JAX में नेटिव Windows GPU सपोर्ट नहीं है।
- **WSL2।** Linux की तरह काम करता है: NVIDIA का Windows ड्राइवर इंस्टॉल करें (WSL के अंदर Linux ड्राइवर नहीं), फिर WSL शेल के अंदर
  Linux वाले कॉलम का पालन करें।
- **पाथ और कोटिंग।** स्पेस वाली किसी भी चीज़ के चारों ओर कोट्स लगाएँ। PowerShell में `curl.exe` कॉल करें (सादा `curl`
  `Invoke-WebRequest` का उपनाम है) और JSON के कोट्स को ऊपर दिखाए अनुसार एस्केप करें।
- **लैपटॉप।** Windows लैपटॉप कुछ मिनट निष्क्रिय रहने पर स्लीप में चले जाते हैं, जिससे लंबा ट्रेनिंग रन रुक जाता है; पावर प्लान बदलें
  या ट्रेनिंग के दौरान मशीन को जागृत रखें।
- **Docker।** Linux पर `docker run --gpus all` के लिए NVIDIA Container Toolkit चाहिए; Windows पर यह Docker
  Desktop के WSL2 बैकएंड के साथ आता है।

## 8. समस्या निवारण

| लक्षण | समाधान |
|---|---|
| `tinyforge: This command needs 'torch'...` | अनुभाग 2 की लाइन से torch इंस्टॉल करें, फिर `tinyforge doctor` दोबारा चलाएँ |
| `incompatible version of torchao` (पुराना tinyforge) | tinyforge अपग्रेड करें, या `pip uninstall -y torchao` |
| `No checkpoint at runs/default/best.pt` | पहले ट्रेन करें (`tinyforge pipeline`), या `--ckpt runs/<preset>/best.pt` दें |
| `cuda_available False` लेकिन आपके पास GPU है | `nvidia-smi` काम करना चाहिए; torch का CUDA बिल्ड इंस्टॉल करें (`cu124` इंडेक्स, `cpu` नहीं) |
| GPU मेमोरी ख़त्म | `tinyforge plan` / `tinyforge ft plan` बताते हैं कि क्या फ़िट होगा; `--max-len` घटाएँ, 4-बिट का उपयोग करें, या `tinyforge memory` देखें |
| `API token not configured` (HTTP 503) | `serve` से पहले `TINYFORGE_API_TOKEN` सेट करें (लोकल टेस्ट के लिए `TINYFORGE_AUTH=off`) |
