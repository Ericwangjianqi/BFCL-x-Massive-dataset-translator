# Translator — LLM-Powered JSON Translation Tool

Translates specific fields inside JSON files from English into any target language using the OpenAI API.

---

## Project Structure

```
translator/
├── data/               ← Put your source JSON files here
├── result/             ← Translated files are written here
├── translate/
│   ├── client.py       ← OpenAI API wrapper (retry logic included)
│   ├── judge.py        ← Gemini-based quality judge
│   └── prompts.py      ← System prompt + user prompt templates
├── config.py           ← Model, fields to translate, batch size
├── main.py             ← Entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

Copy `.env.example` to `.env` and fill in your OpenAI key:

```bash
copy .env.example .env
```

Edit `.env`:
```bash
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIzaSy...  # (Optional) Required only if using the Judge
```

### 3. Configure fields to translate

Open `config.py` and set `TRANSLATE_FIELDS` to the dot-notation paths of the fields you want to translate:

```python
TRANSLATE_FIELDS = [
    "title",
    "description",
    "metadata.summary",   # nested field
]
```

### 4. Configure the Judge (Optional)

You can enable a **Gemini-based Judge** to review and improve translations automatically. The judge checks for:
1. Grammar correctness
2. Naturalness (avoiding stiff/literal translations)
3. Proper noun preservation (file names, paths, code variables, etc.)

To enable it:
1. Add your Gemini API key to `.env`:
   ```bash
   GEMINI_API_KEY=AIzaSy...
   ```
2. In `config.py`, set:
   ```python
   USE_JUDGE = True
   JUDGE_MODEL = "gemini-2.0-flash"  # or "gemini-1.5-pro", etc.
   ```

---

## Usage

```bash
# Translate to Chinese (uses config.py settings)
python main.py --lang Chinese

# Translate to French, override fields via CLI
python main.py --lang French --fields title description

# Auto-detect and translate ALL string fields
python main.py --lang Japanese --fields

# Use a more powerful model with smaller batches
python main.py --lang German --model gpt-4o --batch-size 5
```

### All CLI options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--lang` | `-l` | *(required)* | Target language |
| `--fields` | `-f` | from `config.py` | Fields to translate (dot-notation). Pass with no value to auto-detect. |
| `--model` | `-m` | `config.MODEL` | OpenAI model |
| `--batch-size` | `-b` | `config.BATCH_SIZE` | Texts per API call |
| `--data-dir` | | `data/` | Source directory |
| `--result-dir` | | `result/` | Output directory |

---

## Input / Output Example

**Input** (`data/products.json`):
```json
[
  { "id": 1, "title": "Wireless Headphones", "description": "Great sound quality." },
  { "id": 2, "title": "USB-C Hub",           "description": "7 ports, compact design." }
]
```

**Command:**
```bash
python main.py --lang Chinese
```

**Output** (`result/products.json`):
```json
[
  { "id": 1, "title": "无线耳机", "description": "音质出色。" },
  { "id": 2, "title": "USB-C 集线器", "description": "7个端口，紧凑设计。" }
]
```

Non-translated fields (`id`) are preserved exactly as-is.
