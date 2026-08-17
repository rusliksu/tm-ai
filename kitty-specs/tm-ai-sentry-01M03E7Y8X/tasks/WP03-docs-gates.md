---
work_package_id: "WP03"
title: "Документация и quality gates"
dependencies: ["WP01", "WP02"]
planning_base_branch: "codex/tm-ai-sentry"
merge_target_branch: "main"
phase: "Фаза 3 — Приёмка"
status: "done"
subtasks: ["T005"]
---

# WP03

Зафиксировать переменные окружения и ограничения privacy/deploy, обновить `uv.lock`,
прогнать pytest, ruff, compileall и diff-check.
