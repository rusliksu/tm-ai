---
work_package_id: "WP01"
title: "Reporter и FastAPI boundary"
dependencies: []
planning_base_branch: "codex/tm-ai-sentry"
merge_target_branch: "main"
phase: "Фаза 1 — API"
status: "done"
subtasks: ["T001", "T002"]
---

# WP01

Добавить `sentry-sdk` с env-gated reporter, внедрить его в FastAPI error boundary и
startup path, сохранив существующие HTTP-контракты и локальное логирование.
