#!/bin/bash
# Start a stack of kev.serve backends and the gateway that routes /v1/systemone by the request's
# `model` id (scripts/kev_gateway.py). One backend per checkpoint: kev.serve loads a single model
# per process, so each gets its own port (and its own GPU via CUDA_VISIBLE_DEVICES).
#
# Usage:
#   scripts/serve_stack.sh clef=/path/to/clef-flash@0:8011 \
#                          gui-actor=microsoft/GUI-Actor-7B-Qwen2.5-VL@1:8010 \
#                          [--default clef] [--gateway-port 8020] [--wait 300] [--host HOST] [--stop]
#
#   SPEC = NAME=CKPT[@GPU][:PORT]
#     NAME   the model id clients send, and the gateway's route name
#     CKPT   a checkpoint directory or Hub id kev.serve can load
#     @GPU   CUDA_VISIBLE_DEVICES for that backend (omit = every visible GPU)
#     :PORT  the backend's port (omit = 8008 + the spec's index)
#   (a trailing @REVISION that is not @GPU[:PORT] stays part of CKPT, so owner/name@sha works)
#   HOST (--host) is the interface every server binds: 127.0.0.1 by default, which only answers
#   connections made on THIS machine. A tunnel whose far end dials this machine's loopback works
#   (`ssh -L 8020:127.0.0.1:8020 user@this-box`); anything else that dials another address of it --
#   the box's LAN IP, a container's port map (docker -p), a reverse proxy on another host -- needs
#   `--host 0.0.0.0`, and the clients then use `http://<this machine's IP>:<port>`.
#
# Each backend's log is /tmp/kev-stack-<NAME>.log and the gateway's is /tmp/kev-stack-gateway.log;
# PIDs go to /tmp/kev-stack.pid, which --stop reads. Readiness is the port opening: kev.serve binds
# it only after the model is loaded (3B ~20 s, 4B ~50 s, 7B ~75 s, clef-9B ~90 s), and a backend that
# dies during load fails the script with the tail of its log. The gateway starts once every backend
# answers, and --default (the first NAME when omitted, "" to disable) is what requests naming
# anything else -- e.g. the TypeSafe SDK's "jev-latest" -- are routed to.
set -u
cd "$(dirname "$0")/.."
PROJECT=$PWD
PY=${PYTHON:-$PROJECT/.venv/bin/python}
PIDFILE=/tmp/kev-stack.pid
SPECS=()
DEFAULT=""
DEFAULT_SET=0
GW_PORT=8020
WAIT=300
HOST=127.0.0.1

if [ "${1:-}" = "--stop" ]; then
    if [ -s "$PIDFILE" ]; then
        # shellcheck disable=SC2046
        kill $(cat "$PIDFILE") 2>/dev/null && echo "stopped: $(tr '\n' ' ' < "$PIDFILE")"
        : > "$PIDFILE"
    else
        echo "nothing to stop ($PIDFILE is empty or missing)"
    fi
    exit 0
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --default)       DEFAULT=$2; DEFAULT_SET=1; shift 2 ;;
        --gateway-port)  GW_PORT=$2; shift 2 ;;
        --wait)          WAIT=$2; shift 2 ;;
        --host)          HOST=$2; shift 2 ;;
        --*)             echo "unknown flag $1" >&2; exit 2 ;;
        *)               SPECS+=("$1"); shift ;;
    esac
done
[ ${#SPECS[@]} -gt 0 ] || { echo "give at least one NAME=CKPT[@GPU][:PORT] spec (see the header)" >&2; exit 2; }
STARTED=()
STARTED_PIDS=()   # only this run's processes, so a failure never tears down a stack started elsewhere

fail() {   # tear down what this run started, then report (the backends would otherwise linger half-started)
    echo "$*" >&2
    [ ${#STARTED_PIDS[@]} -gt 0 ] && kill "${STARTED_PIDS[@]}" 2>/dev/null
    exit 1
}

# A port that already answers /v1/models belongs to someone else's server (an older stack, a
# hand-started one). Refuse it up front: kev.serve binds only after loading its model, so a duplicate
# would waste a full load (90 s for clef) before dying on bind -- and the gateway would race an older
# instance for the port, where the probe below can read the old server as success.
in_use() { curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$1/v1/models"; }
# the gateway port first, before any model is loaded
in_use "$GW_PORT" && fail "port $GW_PORT already answers /v1/models (an older gateway?); stop it or give --gateway-port another port"

for i in "${!SPECS[@]}"; do
    spec=${SPECS[$i]}
    name=${spec%%=*}; value=${spec#*=}
    [ "$name" != "$spec" ] && [ -n "$name" ] && [ -n "$value" ] || { echo "bad spec $spec (want NAME=CKPT[@GPU][:PORT])" >&2; exit 2; }
    gpu=""; port=$((8008 + i))
    if   [[ "$value" =~ ^(.*)@([0-9]*):([0-9]+)$ ]]; then ckpt=${BASH_REMATCH[1]}; gpu=${BASH_REMATCH[2]}; port=${BASH_REMATCH[3]}
    elif [[ "$value" =~ ^(.*)@([0-9]+)$ ]];         then ckpt=${BASH_REMATCH[1]}; gpu=${BASH_REMATCH[2]}
    else ckpt=$value; fi

    in_use "$port" && fail "port $port already serves /v1/models; stop that server or give $name another :PORT"

    log=/tmp/kev-stack-$name.log
    echo "starting $name ($ckpt)${gpu:+ on GPU $gpu} port $port; log $log"
    if [ -n "$gpu" ]; then CUDA_VISIBLE_DEVICES=$gpu nohup "$PY" -m kev.serve --run "$ckpt" --port "$port" --host "$HOST" > "$log" 2>&1 &
    else                   nohup "$PY" -m kev.serve --run "$ckpt" --port "$port" --host "$HOST" > "$log" 2>&1 & fi
    pid=$!; echo "$pid" >> "$PIDFILE"; STARTED_PIDS+=("$pid")   # append: another stack's PIDs may be recorded too

    ready=0
    for ((t = 0; t < WAIT; t += 2)); do
        if in_use "$port"; then ready=1; break; fi
        if ! kill -0 "$pid" 2>/dev/null; then
            fail "$name died while loading; tail of $log:
$(tail -5 "$log")"
        fi
        sleep 2
    done
    [ "$ready" = 1 ] || fail "$name did not come up within ${WAIT}s; see $log"
    echo "  $name ready (pid $pid)"
    STARTED+=("$name=http://127.0.0.1:$port")
done

[ "$DEFAULT_SET" = 0 ] && DEFAULT=${SPECS[0]%%=*}
routes=(); for r in "${STARTED[@]}"; do routes+=(--route "$r"); done
log=/tmp/kev-stack-gateway.log
echo "starting gateway on $GW_PORT; log $log"
nohup "$PY" scripts/kev_gateway.py "${routes[@]}" ${DEFAULT:+--default "$DEFAULT"} --port "$GW_PORT" --host "$HOST" > "$log" 2>&1 &
gwpid=$!; echo "$gwpid" >> "$PIDFILE"; STARTED_PIDS+=("$gwpid")
ready=0
for ((t = 0; t < 30; t += 1)); do
    if in_use "$GW_PORT"; then ready=1; break; fi
    if ! kill -0 "$gwpid" 2>/dev/null; then
        fail "gateway died on startup; tail of $log:
$(tail -5 "$log")"
    fi
    sleep 1
done
[ "$ready" = 1 ] || fail "gateway did not answer on $GW_PORT within 30s; see $log"

printf '\n%-12s %-22s %s\n' NAME PORT STATUS
for r in "${STARTED[@]}"; do
    printf '%-12s %-22s %s\n' "${r%%=*}" "${r#*=}" "serving (log /tmp/kev-stack-${r%%=*}.log)"
done
printf '%-12s %-22s %s\n' gateway "http://127.0.0.1:$GW_PORT" "${DEFAULT:+default -> $DEFAULT}"
[ "$HOST" = "127.0.0.1" ] || echo "$(basename "$0"): bound on $HOST: clients elsewhere use http://<this machine's IP>:$GW_PORT (and http://<IP>:<port> for a backend)"
cat <<EOF

Call it (model = a route name):
  curl -s http://127.0.0.1:$GW_PORT/v1/models
  curl -s http://127.0.0.1:$GW_PORT/v1/systemone -H 'content-type: application/json' \\
       -d '{"model": "${SPECS[0]%%=*}", "state": "...", "questions": {"q": {"type": "choice", "instructions": "...", "criteria": {"a": null, "b": null}}}}'

Stop everything:
  scripts/serve_stack.sh --stop        # reads $PIDFILE
EOF
