
## Spezifikation 3: KI‑Server (Python, PyTorch, PPO)

### 3.1 Ziel

- Lokaler KI‑Server, der:
  - über HTTP von deinem TM‑Server aufgerufen wird,  
  - zunächst einfache, aber gültige Züge liefert (Baseline),  
  - aus Logs mit ca. 50 menschlichen Spielen eine initiale Policy lernt (Supervised),  
  - später via Self‑Play‑PPO auf Cloud‑GPUs weitertrainiert werden kann. [en.wikipedia](https://en.wikipedia.org/wiki/Proximal_policy_optimization)

### 3.2 Tech‑Stack

- Sprache: Python 3.11+.  
- Frameworks:
  - Web‑API: FastAPI oder Flask.  
  - DL: PyTorch.  
  - RL: Stable‑Baselines3 (PPO) + eigenes Wrapper‑Environment für TM.  
- Packaging/Deploy:
  - `uv`.  
  - Dockerfile für CPU (lokal) und GPU (Cloud). [synpixcloud](https://www.synpixcloud.com/blog/cloud-gpu-pricing-comparison-2026)

### 3.3 API‑Design

**Endpoint: `POST /move`**

- Request:
  - Body wie oben (State + legal_actions).  
- Response:
  - `action_id`, `parameters`, optional `debug`.  

**Weitere Endpoints:**

- `GET /health`: `{ "status": "ok" }`.  
- `GET /version`: Commit‑Hash, Modell‑Version, Model‑Config.  
- Optional: `POST /train_step` (nur intern genutzt, wenn du Training „online“ machen willst – für den Anfang nicht notwendig).  

### 3.4 State‑Encoding & Aktion‑Schema

**State‑Encoder:**

- Modul `encoding.py`:
  - Nimmt JSON‑State des TM‑Servers und erzeugt:
    - `state_vector` (NumPy/Torch‑Tensor):  
      - globale Features (Temperatur, Ozeane, O2, Generation, Milestones/Awards, global tags counts).  
      - pro Spieler: TR, Ressourcen, Produktionen, Anzahl Tags, wichtige Karten‑Features aggregiert.  
      - aktueller Spieler: zusätzliche Features (Handkartenzahl, Durchschnittskosten etc.).  
    - `action_mask`: Binärvektor `len(ACTION_SPACE)` lang, der anzeigt, welche abstrakten Aktionen erlaubt sind.  

**Action‑Schema:**

- Zunächst abstraktes, reduziertes Aktionsset, z.B.:
  - „Spiele Karte i“ (auf Slots gemappt),  
  - „Standardprojekt X“,  
  - „Passe“.  
- Mapping‑Modul, das zwischen diesem abstrakten Schema und den `legal_actions` des TM‑Servers vermittelt.  

### 3.5 Modell & Trainingspipeline

**Modell (erste Version):**

- `PolicyValueNet` (MLP):
  - Input: `state_vector`.  
  - 3–4 Hidden‑Layer à 256–512 Neuronen, ReLU, LayerNorm, Dropout.  
  - Policy‑Head: logits über `ACTION_SPACE`, Softmax + Maskierung mit `action_mask`.  
  - Value‑Head: skalarer State‑Wert \(v(s)\).  

**Supervised‑Training (Phase 1):**

- Daten: 
  - Aus deinen existierenden Logs die menschlichen Züge extrahieren.  
- Loss:
  - Cross‑Entropy zwischen Policy‑Head und „gewählter Aktion“ (ggf. gewichtet nach Endergebnissen).  
  - MSE oder L2‑Loss für Value‑Head gegen finalen Reward (z.B. normalisierte TR/Platzierung).  
- Training:
  - CPU‑Training ausreichend, evtl. 10–50 Epochen, Batch‑Size 64–256.  

**PPO‑Training (Phase 2, in der Cloud):**

- TM‑Environment:
  - Python‑Wrapper, der über HTTP einen TM‑Server steuert:
    - `reset()` startet neues Spiel,  
    - `step(action)` schickt Aktion für aktuellen Spieler, liest neuen State + Reward.  
- Stable‑Baselines3:
  - PPO mit MLP‑Policy, ggf. initialisiert mit deinem vortrainierten Policy‑Netz.  
  - Hyperparameter (Startwerte):
    - `learning_rate = 3e-4`,  
    - `gamma = 0.99`,  
    - `gae_lambda = 0.95`,  
    - `n_steps` so wählen, dass pro Update einige tausend Schritte gesammelt werden,  
    - `clip_range = 0.2`. [huggingface](https://huggingface.co/blog/deep-rl-ppo)

### 3.6 Lokale Nutzung vs. Cloud‑Training

**Lokal (Laptop ohne GPU):**

- KI‑Server:
  - Läuft im „Inferenz‑Modus“ mit CPU, evtl. kleiner Batch‑Size.  
  - Training nur für kleine Tests (z.B. 1–2 Epochen auf wenigen Samples).  

**Cloud (RunPod/Vast/Synpix):**

- Docker‑Image:
  - Basis‑Image mit GPU‑Support (z.B. `pytorch/pytorch` + CUDA).  
  - Enthält KI‑Server + Option, TM‑Server im gleichen Container oder per Compose‑Netzwerk.  
- Startscripts:
  - `run_inference.sh` (nur KI‑Server, für spätere Online‑Spiele).  
  - `run_training.sh` (startet TM‑Server + Self‑Play‑Training mit PPO).  
- Kostenkontrolle:
  - Trainingsjobs so schreiben, dass sie nach N Millionen Schritten sauber stoppen und Ergebnisse (Models, Logs) in ein Cloud‑Volume oder S3‑kompatiblen Storage schreiben. [synpixcloud](https://www.synpixcloud.com/ko/blog/cloud-gpu-pricing-comparison-2026)

Hier ist ein konkretisiertes API‑Schema plus ein schlanker Skeleton‑Code für deinen KI‑Server mit FastAPI und `uv` als Paketmanager. [huggingface](https://huggingface.co/blog/deep-rl-ppo)


## API‑Schema

### Basis

- Base URL: `http://localhost:8000`
- Content‑Type: `application/json`
- Endpoints:
  - `POST /move` – Kern‑API für den TM‑Server
  - `GET /health` – Healthcheck
  - `GET /version` – Meta‑Infos (Modellversion etc.)

### `POST /move`

**Request‑Body (vom TM‑Server):**

```json
{
  "game_id": "tm-2025-05-01-001",
  "player_id": "p2",
  "state": {
    "global": {
      "generation": 7,
      "temperature": -12,
      "oxygen": 8,
      "oceans": 5
    },
    "players": [
      {
        "player_id": "p1",
        "tr": 42,
        "resources": {
          "megacredits": 25,
          "steel": 3,
          "titanium": 1,
          "plants": 5,
          "energy": 2,
          "heat": 6
        },
        "production": {
          "megacredits": 4,
          "steel": 1,
          "titanium": 0,
          "plants": 2,
          "energy": 1,
          "heat": 0
        },
        "tags": {
          "science": 2,
          "building": 3,
          "space": 1
        }
      }
    ],
    "current_player_id": "p2",
    "hand": [
      {
        "card_id": "card_123",
        "cost": 13,
        "tags": ["science"],
        "requirements": {
          "min_temperature": -20
        }
      }
    ],
    "phase": "action_phase"
  },
  "legal_actions": [
    {
      "action_id": "play_card",
      "params": {
        "card_id": "card_123"
      }
    },
    {
      "action_id": "standard_project_heat_to_temp"
    },
    {
      "action_id": "pass"
    }
  ],
  "metadata": {
    "schema_version": 1
  }
}
```

**Response‑Body (vom KI‑Server):**

```json
{
  "action_id": "play_card",
  "parameters": {
    "card_id": "card_123"
  },
  "debug": {
    "policy_logits": [1.2, -0.3, 0.1],
    "value_estimate": 0.35
  }
}
```

- `action_id`: muss einer der `legal_actions[*].action_id` sein.  
- `parameters`: optional, nur wenn für diese Aktion notwendig.  
- `debug`: nur für Logging/Analyse, der TM‑Server kann es ignorieren.  

### `GET /health`

**Response:**

```json
{
  "status": "ok"
}
```

### `GET /version`

**Response:**

```json
{
  "model_version": "0.1.0",
  "git_commit": "abc123def",
  "config": {
    "state_dim": 512,
    "hidden_sizes": [512, 512, 512],
    "action_space_size": 64
  }
}
```

***

## Projektstruktur mit `uv`

Vorschlag:

```text
tm-ai-server/
  pyproject.toml
  uv.lock
  src/
    tm_ai_server/
      __init__.py
      main.py           # FastAPI App, Endpoints
      schemas.py        # Pydantic-Modelle für Request/Response
      encoding.py       # TM-JSON -> Tensoren / Masken
      model.py          # PolicyValueNet (PyTorch)
      inference.py      # Laden des Modells, Auswahl der Aktion
      config.py         # Pfade, Hyperparameter, Defaults
      training/
        __init__.py
        dataset.py      # Laden der Log-Dateien
        train_supervised.py
        train_ppo_env.py
        env_tm.py       # RL-Environment für PPO (später)
```

**Init mit uv (nur Info, kein Code):**

```bash
uv init tm-ai-server
cd tm-ai-server
uv add fastapi uvicorn[standard] pydantic torch numpy
# später:
uv add "stable-baselines3[extra]"

***

## Review Remarks
- Die Beispiel-API zeigt `state.global`, aber das Pydantic-Modell nutzt `GameState.global_`; dieser Feldname-Mismatch muss geklärt oder über Aliase gelöst werden.
- Die `encoding.py`-Skeleton enthält fehlerhafte Markdown-Link-Texte (`[huggingface]`, `[datacamp]`) innerhalb des Codeblockes, was unbrauchbaren Beispielcode erzeugt.
- `action_space_size = 64` wird genannt, aber die konkrete Definition und das Mapping auf reale legale Aktionen fehlen.
- Es fehlen Spezifikationen für die Belohnungsfunktion im PPO-Training, insbesondere wie finaler Score / TR / Platzierung in einen numerischen Reward übersetzt wird.
- Es wird ein optionaler `POST /train_step`-Endpoint erwähnt, aber nicht im API-Schema definiert.
- Die vorgeschlagene Projektstruktur listet Trainingsmodule, aber es fehlen konkrete Datenformate für `dataset.py` und `env_tm.py`.
- Die Dokumentation sollte zusätzlich ein Beispiel `Dockerfile` / `docker-compose.yml` zur besseren Umsetzbarkeit enthalten.
- `uv` ist ein spezifischer Paketmanager; falls das Projekt später auf `pip` oder `poetry` umsteigt, sollte das dokumentiert werden.

## Detailed Inconsistencies and Gaps
- `MoveRequest` im Schema erwartet `GameState.global_`, aber Beispiel-JSON nutzt `global`; das führt zu Deserialisierungsfehlern.
- Die `build_action_space`-Funktion erzeugt aktuell nur eine 1:1-Abbildung und keine echte abstrakte Aktions-Maske.
- Das Modell-Skeleton verwendet `List[int]` für `hidden_sizes`, aber der Default ist ein Tuple; das ist zwar praktisch, sollte jedoch konsistent sein.
- Für Produktionsreife fehlen Angaben zur Modellpersistenz (`save`/`load`), zur Versionskompatibilität der Checkpoints und zu einer möglichen `model_version`-Strategie.
- Es gibt keine Bewertung oder Überwachung der Laufzeit/Performance der API, was bei späterem Cloud-Einsatz wichtig ist.
```

***

## Skeleton‑Code

### `src/tm_ai_server/schemas.py`

```python
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class GlobalState(BaseModel):
    generation: int
    temperature: int
    oxygen: int
    oceans: int


class PlayerResources(BaseModel):
    megacredits: int
    steel: int
    titanium: int
    plants: int
    energy: int
    heat: int


class PlayerProduction(BaseModel):
    megacredits: int
    steel: int
    titanium: int
    plants: int
    energy: int
    heat: int


class PlayerTags(BaseModel):
    science: int = 0
    building: int = 0
    space: int = 0


class PlayerState(BaseModel):
    player_id: str
    tr: int
    resources: PlayerResources
    production: PlayerProduction
    tags: PlayerTags


class CardRequirement(BaseModel):
    min_temperature: Optional[int] = None
    max_temperature: Optional[int] = None
    # später erweitern (O2, Ozeane, Tags, etc.)


class CardState(BaseModel):
    card_id: str
    cost: int
    tags: List[str] = []
    requirements: Optional[CardRequirement] = None


class GameState(BaseModel):
    global_: GlobalState
    players: List[PlayerState]
    current_player_id: str
    hand: List[CardState] = []
    phase: str


class LegalAction(BaseModel):
    action_id: str
    params: Dict[str, Any] = {}


class Metadata(BaseModel):
    schema_version: int = 1


class MoveRequest(BaseModel):
    game_id: str
    player_id: str
    state: GameState
    legal_actions: List[LegalAction]
    metadata: Metadata


class MoveDebug(BaseModel):
    policy_logits: Optional[List[float]] = None
    value_estimate: Optional[float] = None


class MoveResponse(BaseModel):
    action_id: str
    parameters: Dict[str, Any] = {}
    debug: Optional[MoveDebug] = None


class HealthResponse(BaseModel):
    status: str = "ok"


class VersionConfig(BaseModel):
    state_dim: int
    hidden_sizes: List[int]
    action_space_size: int


class VersionResponse(BaseModel):
    model_version: str
    git_commit: str
    config: VersionConfig
```

### `src/tm_ai_server/config.py`

```python
import os
from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    state_dim: int = 512
    hidden_sizes: List[int] = (512, 512, 512)
    action_space_size: int = 64
    model_path: str = os.getenv("MODEL_PATH", "models/policy_value.pt")


@dataclass
class AppConfig:
    model_version: str = os.getenv("MODEL_VERSION", "0.1.0")
    git_commit: str = os.getenv("GIT_COMMIT", "dev")
    model: ModelConfig = ModelConfig()
```

### `src/tm_ai_server/model.py`

```python
from typing import Tuple

import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_sizes, action_space_size: int):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.backbone = nn.Sequential(*layers)
        self.policy_head = nn.Linear(input_dim, action_space_size)
        self.value_head = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value
```

### `src/tm_ai_server/encoding.py`

```python
from typing import List, Tuple

import numpy as np
import torch

from .schemas import GameState, LegalAction


def encode_state(game_state: GameState) -> np.ndarray:
    # Platzhalter – hier später echte Features bauen
    # z.B. globale Werte, pro Spieler aggregierte Stats, Hand-Features etc.
    # Jetzt nur Dummy-Vektor, damit der Skeleton kompilierbar ist.
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = game_state.global_.generation
    vec [huggingface](https://huggingface.co/blog/deep-rl-ppo) = game_state.global_.temperature
    vec [datacamp](https://www.datacamp.com/tutorial/proximal-policy-optimization) = game_state.global_.oxygen
    vec[3] = game_state.global_.oceans
    # TODO: weitere Features einbauen
    return vec


def build_action_space(legal_actions: List[LegalAction]) -> Tuple[List[LegalAction], torch.Tensor]:
    """
    Placeholder: im Moment ist ACTION_SPACE == legal_actions.
    Später: globaler Action-Space (z.B. 64 Slots) + Maske.
    """
    n = len(legal_actions)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:] = True
    return legal_actions, mask
```

### `src/tm_ai_server/inference.py`

```python
from typing import List, Tuple

import torch

from .config import AppConfig
from .encoding import encode_state, build_action_space
from .model import PolicyValueNet
from .schemas import GameState, LegalAction


class InferenceEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.device = torch.device("cpu")  # später GPU optional
        self.model = PolicyValueNet(
            state_dim=config.model.state_dim,
            hidden_sizes=config.model.hidden_sizes,
            action_space_size=config.model.action_space_size,
        ).to(self.device)
        self.model.eval()
        self._load_weights_if_available(config.model.model_path)

    def _load_weights_if_available(self, path: str) -> None:
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        except FileNotFoundError:
            # Für den Anfang einfach uninitialisiertes Netz benutzen
            pass

    def select_action(
        self,
        state: GameState,
        legal_actions: List[LegalAction],
    ) -> Tuple[LegalAction, torch.Tensor, float]:
        state_vec = encode_state(state)
        state_tensor = torch.from_numpy(state_vec).unsqueeze(0).to(self.device)

        action_space, action_mask = build_action_space(legal_actions)
        logits, value = self.model(state_tensor)

        # Auf legale Aktionen einschränken (hier: simple Variante)
        logits = logits.squeeze(0)
        if logits.shape[0] != len(action_space):
            # Fallback: erste legale Aktion
            return legal_actions[0], logits.detach().cpu(), float(value.item())

        masked_logits = logits.clone()
        masked_logits[~action_mask] = -1e9
        probs = torch.softmax(masked_logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()

        chosen_action = action_space[idx]
        return chosen_action, logits.detach().cpu(), float(value.item())
```

### `src/tm_ai_server/main.py`

```python
from fastapi import FastAPI, HTTPException

from .config import AppConfig
from .inference import InferenceEngine
from .schemas import (
    HealthResponse,
    MoveRequest,
    MoveResponse,
    MoveDebug,
    VersionResponse,
    VersionConfig,
)

app = FastAPI(title="TM AI Server", version="0.1.0")

config = AppConfig()
engine = InferenceEngine(config=config)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/version", response_model=VersionResponse)
async def version() -> VersionResponse:
    return VersionResponse(
        model_version=config.model_version,
        git_commit=config.git_commit,
        config=VersionConfig(
            state_dim=config.model.state_dim,
            hidden_sizes=list(config.model.hidden_sizes),
            action_space_size=config.model.action_space_size,
        ),
    )


@app.post("/move", response_model=MoveResponse)
async def move(req: MoveRequest) -> MoveResponse:
    if not req.legal_actions:
        raise HTTPException(status_code=400, detail="No legal_actions provided")

    action, logits, value = engine.select_action(req.state, req.legal_actions)
    debug = MoveDebug(
        policy_logits=logits.tolist(),
        value_estimate=value,
    )
    return MoveResponse(
        action_id=action.action_id,
        parameters=action.params,
        debug=debug,
    )
```

**Start (lokal):**

```bash
uv run uvicorn tm_ai_server.main:app --reload --host 0.0.0.0 --port 8000
```

***
