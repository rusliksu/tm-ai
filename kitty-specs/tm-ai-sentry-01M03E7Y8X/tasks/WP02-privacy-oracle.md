---
work_package_id: "WP02"
title: "Privacy oracle и regression tests"
dependencies: ["WP01"]
planning_base_branch: "codex/tm-ai-sentry"
merge_target_branch: "main"
phase: "Фаза 2 — Проверки"
status: "done"
subtasks: ["T003", "T004"]
---

# WP02

Проверить фактический fake envelope и негативные sentinels; убедиться, что ожидаемые
4xx, health и version не создают шумовых событий.
