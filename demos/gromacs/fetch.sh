#!/usr/bin/env bash
# Fetch GROMACS benchmark archives into demos/gromacs/out/benchmarks.
#
#   bash demos/gromacs/fetch.sh                    # the seven advertised sets
#   bash demos/gromacs/fetch.sh water_1.5k-6.1M    # named archives
#   bash demos/gromacs/fetch.sh --list             # everything the origin holds
#
# The gitlab.io download links redirect to a KTH PDC Swift container, which
# throttles per connection at about 65 kB/s -- 510 MB would take two hours. It
# does honour Range requests, and the throttle is per connection rather than per
# client, so this splits each archive into N chunks fetched concurrently: 8 ways
# measured 559 kB/s against 67 kB/s serial on the same file. Chunks are
# reassembled in order and the result is gzip-tested, because a truncated
# member of a tarball otherwise surfaces much later as a confusing extract error.
set -uo pipefail

ORIGIN="https://s3.dc.pdc.kth.se/swift/v1/AUTH_fb28ce6f151e4b5ea5639956e43f61ed/rnd-benchmarks"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/out/benchmarks"
JOBS="${CUMETAL_FETCH_JOBS:-8}"

if [[ "${1:-}" == "--list" ]]; then
    curl -sfL --max-time 60 "${ORIGIN}"; exit
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then sed -n '2,14p' "$0"; exit 0; fi

ARCHIVES=("$@")
[[ ${#ARCHIVES[@]} -gt 0 ]] || ARCHIVES=(villin rnase ADH stmv_gmx_v2 water_1.5k-6.1M grappa_1.5k-6.1M)
mkdir -p "${DEST}"

for name in "${ARCHIVES[@]}"; do
    name="${name%.tar.gz}"
    out="${DEST}/${name}.tar.gz"
    if gzip -t "${out}" 2>/dev/null; then echo "have ${name}.tar.gz"; continue; fi
    size="$(curl -sfI --max-time 60 "${ORIGIN}/${name}.tar.gz" | tr -d '\r' |
            sed -n 's/^[Cc]ontent-[Ll]ength: //p' | tail -1)"
    if [[ -z "${size}" || "${size}" -le 0 ]]; then echo "FAIL: no ${name}.tar.gz at origin"; continue; fi
    echo "fetching ${name}.tar.gz ($(( size / 1048576 )) MiB, ${JOBS} connections) ..."

    tmp="${out}.parts"; mkdir -p "${tmp}"
    chunk=$(( (size + JOBS - 1) / JOBS ))
    # A chunk can come back short without curl failing, and one short chunk
    # ruins the whole reassembly -- so each pass re-fetches only the chunks that
    # are not exactly the length their range asked for. Complete chunks survive
    # across passes and across runs of this script, which makes it resumable.
    for pass in 1 2 3 4 5; do
        missing=0
        for ((i = 0; i < JOBS; i++)); do
            lo=$(( i * chunk )); hi=$(( lo + chunk - 1 ))
            (( hi >= size )) && hi=$(( size - 1 ))
            (( lo > hi )) && continue
            part="${tmp}/$(printf '%03d' "${i}")"
            want=$(( hi - lo + 1 ))
            [[ -f "${part}" && "$(stat -f%z "${part}" 2>/dev/null || echo 0)" == "${want}" ]] && continue
            missing=$(( missing + 1 ))
            curl -sfL --retry 5 --retry-all-errors --max-time 3600 \
                 -r "${lo}-${hi}" -o "${part}" "${ORIGIN}/${name}.tar.gz" &
        done
        wait
        (( missing == 0 )) && break
        [[ ${pass} -gt 1 ]] && echo "  pass ${pass}: refetched ${missing} short chunk(s)"
    done

    cat "${tmp}"/* > "${out}" && rm -rf "${tmp}"
    got="$(stat -f%z "${out}" 2>/dev/null || echo 0)"
    if [[ "${got}" != "${size}" ]] || ! gzip -t "${out}" 2>/dev/null; then
        echo "FAIL: ${name}.tar.gz is ${got} of ${size} bytes, or not valid gzip"; rm -f "${out}"; continue
    fi
    echo "  ok, extracting"
    tar xzf "${out}" -C "${DEST}"
done
