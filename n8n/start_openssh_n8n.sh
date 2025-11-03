#!/usr/bin/env bash
set -euo pipefail

# === 🔧 CONFIG: Pfade und Service-Namen anpassen ===
# Projektverzeichnis mit deiner docker-compose.yml
PROJECT_DIR="C:/Projects/DataScience/Portfolio/Turbine-Maintenance-ETL/Workspace"
DOCKER_DATA_DIR="D:/Projects/Docker/n8n"
COMPOSE_FILE="%DOCKER_DATA_DIR%/docker-compose.yml"
# Compose-Service-Name für n8n (wie im compose definiert)
N8N_SERVICE="n8n"
# Docker Desktop EXE (Standard-Installationspfad)
DOCKER_DESKTOP_EXE="C:/Program Files/Docker/Docker/Docker Desktop.exe"
# n8n URL
N8N_URL="http://localhost:5678"

# === 🧠 Hilfsfunktionen ===
winpath() { cygpath -w "$1"; }   # -> Windows-Pfad
unixpath() { cygpath -u "$1"; }  # -> /c/... Pfad

log() { echo "[$(date '+%F %T')] $*"; }

start_sshd() {
  log "🔐 Starte OpenSSH-Server (sshd) …"
  # Versuche zu starten; ignoriere Fehler, falls schon läuft
  powershell.exe -NoProfile -Command "Try { Start-Service sshd } Catch { }" >/dev/null 2>&1 || true
  # Status prüfen
  if powershell.exe -NoProfile -Command "(Get-Service sshd).Status" | grep -qi running; then
    log "🔐 sshd läuft."
  else
    log "⚠️  sshd konnte nicht gestartet werden (evtl. Admin-Rechte erforderlich)."
  fi
}

is_docker_up() {
  docker info >/dev/null 2>&1
}

start_docker_desktop() {
  if is_docker_up; then
    log "🐳 Docker ist bereits verfügbar."
    return 0
  fi
  log "🐳 Starte Docker Desktop …"
  # Docker Desktop im Hintergrund starten
  cmd.exe /c start "" "$(winpath "$DOCKER_DESKTOP_EXE")" >/dev/null 2>&1 || true

  # Warten bis der Docker Daemon bereit ist
  log "⏳ Warte auf Docker Daemon …"
  for i in {1..60}; do
    if is_docker_up; then
      log "🐳 Docker ist bereit."
      return 0
    fi
    sleep 2
  done
  log "❌ Docker wurde nicht rechtzeitig bereit. Bitte Docker Desktop prüfen."
  exit 1
}

compose_up_n8n() {
  local compose_u=$(unixpath "$COMPOSE_FILE")
  local proj_dir=$(unixpath "$PROJECT_DIR")

  if [[ ! -f "$compose_u" ]]; then
    log "❌ compose-Datei nicht gefunden: $COMPOSE_FILE"
    exit 1
  fi

  log "📦 docker compose up -d ($N8N_SERVICE) in $PROJECT_DIR"
  pushd "$proj_dir" >/dev/null

  # Nur n8n-Service hochfahren (falls andere Services existieren, bleiben unberührt)
  docker compose -f "$compose_u" up -d "$N8N_SERVICE"

  popd >/dev/null

  # Check: läuft der Container?
  if docker ps --filter "name=$N8N_SERVICE" --format '{{.Names}}' | grep -q "$N8N_SERVICE"; then
    log "✅ $N8N_SERVICE ist gestartet."
  else
    log "⚠️  $N8N_SERVICE scheint nicht zu laufen. Logs:"
    docker compose -f "$compose_u" logs --tail=100 "$N8N_SERVICE" || true
    exit 1
  fi
}

open_n8n() {
  log "🌐 Öffne $N8N_URL …"
  cmd.exe /c start "" "$N8N_URL" >/dev/null 2>&1 || true
}

# === 🚀 Ablauf ===
start_sshd
start_docker_desktop
compose_up_n8n
open_n8n

log "🎉 Stack bereit."
