#!/usr/bin/env bash
set -Eeuo pipefail

: "${BOOT_MODE:="windows"}"

APP="OmniParser"
SUPPORT="https://github.com/microsoft/OmniParser"

cd /run

. reset.sh      # Initialize system
. define.sh     # Define versions
. install.sh    # Run installation
. disk.sh       # Initialize disks
. display.sh    # Initialize graphics
. network.sh    # Initialize network
. samba.sh      # Configure samba
. boot.sh       # Configure boot
. proc.sh       # Initialize processor
. power.sh      # Configure shutdown
. config.sh     # Configure arguments

trap - ERR

if [ -d "/usr/share/novnc" ]; then
    info "Starting noVNC web interface on port 8007..."
    # Wir verbinden den Web-Port 8007 mit dem VNC-Port 5900 von QEMU
    websockify --web /usr/share/novnc/ 8007 localhost:5900 >/dev/null 2>&1 &
else
    warn "noVNC directory not found, web interface will be disabled."
fi

version=$(qemu-system-x86_64 --version | head -n 1 | cut -d '(' -f 1 | awk '{ print $NF }')
info "Booting ${APP}${BOOT_DESC} using QEMU v$version..."

{ qemu-system-x86_64 ${ARGS:+ $ARGS} >"$QEMU_OUT" 2>"$QEMU_LOG"; rc=$?; } || :
(( rc != 0 )) && error "$(<"$QEMU_LOG")" && exit 15

terminal
( sleep 30; boot ) &
tail -fn +0 "$QEMU_LOG" 2>/dev/null &
cat "$QEMU_TERM" 2> /dev/null | tee "$QEMU_PTY" &
wait $! || :

sleep 1 & wait $!
[ ! -f "$QEMU_END" ] && finish 0
