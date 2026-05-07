## Spezifikation 1: Gesamtvorhaben

### Ziele

- KI‑Agent, der Terraforming Mars in der Open‑Source‑Online‑Umsetzung spielen kann.  
- Fokus: Machbarkeit, verständliche Architektur, nachvollziehbares Training; spielerische Qualität kann durch zusätzliches Training verbessert werden.  
- Training zunächst lokal (Laptop ohne GPU) mit kleinen Experimenten, dann ernsthafte Trainingsläufe auf günstigen GPU‑Providern (RunPod/Vast/Synpix) per Docker. [synpixcloud](https://www.synpixcloud.com/blog/cloud-gpu-pricing-comparison-2026)

### Systemarchitektur (High‑Level)

Komponenten:

1. **TM‑Server (angepasster Fork):**
   - Node/TypeScript‑Server, wie im Originalrepo, lokal oder in der Cloud deployt. [reddit](https://www.reddit.com/r/boardgames/comments/961yjz/terraforming_mars_inventrix_adaptation_technology/)
   - Erweiterungen:
     - AI‑Spielertyp („AI_PLAYER“) im Match‑Setup.  
     - Logging von Spielzuständen und Aktionen (JSON) für Training. [reddit](https://www.reddit.com/r/boardgames/comments/1hl9hke/opensource_version_of_terraforming_mars/)
     - HTTP‑Client, der bei Zug eines AI‑Spielers den KI‑Server aufruft und dessen Aktion ausführt.  

2. **KI‑Server (Python, PyTorch, Stable‑Baselines3):**
   - REST‑API (`/move`, `/health`, `/version`), optional `/train`.  
   - Modul für State‑Encoding (TM‑JSON → Tensoren).  
   - Policy/Value‑Netz (erst einfaches MLP), später ggf. AlphaZero‑light.  
   - Trainingspipeline:
     - Supervised Learning auf ca. 50 vorhandenen menschlichen Partien (lokale TM‑Daten).  
     - Self‑Play‑PPO (oder AlphaZero‑light) über den TM‑Server als Environment.  

3. **Trainings‑/Deployment‑Umgebung:**
   - **Lokal:**  
     - TM‑Server auf Laptop (nur CPU).  
     - KI‑Server lokal (CPU‑Training sehr klein, primär Inferenz und Debug).  
   - **Cloud (RunPod/Vast/Synpix):**  
     - Docker‑Image, das KI‑Server + optional TM‑Server enthält. [synpixcloud](https://www.synpixcloud.com/ko/blog/cloud-gpu-pricing-comparison-2026)
     - GPU‑Instanz (z.B. RTX 4090 oder A100), auf der Self‑Play + PPO‑Training laufen. [gpucloudlist](https://www.gpucloudlist.com/en/blog/google-cloud-gpu-pricing-guide)

### Nicht‑funktionale Anforderungen

- **Portabilität:**  
  - Ganzer Stack (TM‑Server + KI‑Server) über Docker compose startbar, sowohl lokal (CPU) als auch auf GPU‑Instanzen.  
- **Konfigurierbarkeit:**  
  - Endpoints, Ports und DB‑Backends über Environment‑Variablen konfigurierbar. [github](https://github.com/Gugatec/terraforming-mars_deuteranopia-colors)
- **Reproduzierbarkeit:**  
  - Trainingsskripte mit Seeds und klaren Config‑Dateien (YAML/JSON).  
- **Sicherheit:**  
  - KI‑Server nur lokal bzw. im privaten Netzwerk, keine offene Internet‑API.  

## Review Remarks
- Das Architekturmodell ist klar, aber es fehlen konkrete Schnittstellen- und API-Verträge zwischen TM‑Server und KI‑Server.
- Es gibt keine Angabe zur Datenformat-Spezifikation der Trainingslogs oder der erwarteten Log-Struktur.
- Bewertungsmetriken und Erfolgskriterien für den KI-Agenten werden nicht definiert.
- Die Cloud-Deployment-Beschreibung sollte Netzwerksicherheitsanforderungen genauer ausführen, damit „nur privates Netzwerk" bei GPU-Deployments wirklich eingehalten wird.
- Für Reproduzierbarkeit fehlen Hinweise zu Versionierung, Modell-Checkpointing und experimentellen Metadaten.

