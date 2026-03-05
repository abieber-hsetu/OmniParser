#!/usr/bin/env bash
set -Eeuo pipefail

: "${SAMBA:="Y"}"

[[ "$SAMBA" == [Nn]* ]] && return 0
[[ "$NETWORK" == [Nn]* ]] && return 0

# Exakte Übereinstimmung mit deiner Vorgabe
hostname="OMNIBOX"
interface="eth0 lo"

# Wir deaktivieren die DHCP-Überschreibung für den Hostnamen, 
# damit es stabil bei OMNIBOX bleibt
if [[ "$DHCP" == [Yy1]* ]]; then
  interface="$VM_NET_DEV"
fi

addShare() {
  local dir="$1"
  local name="$2"
  local comment="$3"

  mkdir -p "$dir" || return 1

  if [ -z "$(ls -A "$dir")" ]; then
    chmod 777 "$dir"
    {      echo "--------------------------------------------------------"
           echo " $APP"
           echo " For support visit $SUPPORT"
           echo "--------------------------------------------------------"
           echo ""
           echo "Verzeichnis für den Datenaustausch mit dem Host."
    } | unix2dos > "$dir/readme.txt"
  fi

  # Share-Konfiguration angepasst an deine Vorgabe
  {       echo ""
          echo "[$name]"
          echo "    path = $dir"
          echo "    comment = $comment"
          echo "    browseable = yes"
          echo "    read only = no"
          echo "    guest ok = yes"
          echo "    force user = root"
  } >> "/etc/samba/smb.conf"

  return 0
}

# Global-Sektion exakt nach deiner Vorgabe
{       echo "[global]"
        echo "    workgroup = WORKGROUP"
        echo "    server string = OmniBox Share"
        echo "    netbios name = $hostname"
        echo "    security = user"
        echo "    map to guest = Bad User"
        echo "    interfaces = $interface"
        echo "    bind interfaces only = no" # Wichtig für die Erreichbarkeit
        echo "    server min protocol = NT1"
        echo ""
        echo "    # disable printing services"
        echo "    load printers = no"
        echo "    printing = bsd"
        echo "    printcap name = /dev/null"
        echo "    disable spoolss = yes"
} > "/etc/samba/smb.conf"

# Pfad auf dein gewünschtes Verzeichnis setzen
share="/run/import"

addShare "$share" "Data" "Shared" || error "Failed to create shared folder!"

# Start des Daemons
if ! smbd; then
  error "Samba daemon failed to start!"
  smbd -i --debug-stdout || true
fi

# Web Service Discovery (für Windows 10/11 Sichtbarkeit)
if [[ "${BOOT_MODE:-}" != "windows_legacy" ]]; then
  wsdd -i "$interface" -p -n "$hostname" &
  echo "$!" > /var/run/wsdd.pid
fi

return 0