#!/usr/bin/env bash
set -euo pipefail

compiler=${1:?usage: run_cumetalc_nvcc_options_file_test.sh <cumetalc> <work-dir>}
work=${2:?}
rm -rf "${work}"
mkdir -p "${work}/include path" "${work}/second path"
cat > "${work}/include path/value.h" <<'HEADER'
#define INCLUDED_VALUE 19
HEADER
cat > "${work}/second path/system_value.h" <<'HEADER'
#define SYSTEM_VALUE 23
HEADER
cat > "${work}/main.cu" <<'SOURCE'
#include <cstdio>
#include <value.h>
#include <system_value.h>
int main() {
    const int actual = INCLUDED_VALUE + SYSTEM_VALUE + FILE_VALUE + COMMAND_VALUE;
    std::printf("options value=%d\n", actual);
    return actual == 70 ? 0 : 1;
}
SOURCE
cat > "${work}/includes.rsp" <<OPTIONS
-I"${work}/include path"
-I '${work}/second path'
-DFILE_VALUE=11
OPTIONS
cat > "${work}/empty.rsp" <<'OPTIONS'
OPTIONS
cat > "${work}/nested.rsp" <<OPTIONS
--options-file "${work}/includes.rsp"
OPTIONS

nvcc() { CUMETAL_NVCC_COMPAT=1 "${compiler}" "$@"; }
for spelling in --options-file -optf; do
    for form in separate equals; do
        option=("${spelling}" "${work}/empty.rsp,${work}/nested.rsp")
        if [[ ${form} == equals ]]; then
            option=("${spelling}=${work}/empty.rsp,${work}/nested.rsp")
        fi
        nvcc "${work}/main.cu" -DCOMMAND_VALUE=17 "${option[@]}" \
            -c -o "${work}/main.o"
        nvcc "${work}/main.o" -o "${work}/main"
        [[ "$("${work}/main")" == 'options value=70' ]]
    done
done

# A response file must remain data, including shell metacharacters and escapes.
mkdir -p "${work}/literal\$(false) path"
cp "${work}/include path/value.h" "${work}/literal\$(false) path/value.h"
cat > "${work}/escaped.rsp" <<OPTIONS
-I${work}/literal\$(false)\ path
-I"${work}/second path"
-DFILE_VALUE=11
OPTIONS
nvcc "${work}/main.cu" -DCOMMAND_VALUE=17 --options-file "${work}/escaped.rsp" \
    -o "${work}/main"
[[ "$("${work}/main")" == 'options value=70' ]]

expect_error() {
    local expected=$1
    shift
    local status=0
    nvcc "$@" > "${work}/error.log" 2>&1 || status=$?
    [[ ${status} == 2 ]] || { cat "${work}/error.log"; exit 1; }
    grep -Fq -- "${expected}" "${work}/error.log"
}
printf '%s\n' '"unterminated' > "${work}/quote.rsp"
printf '%s' '-I dangling\' > "${work}/escape.rsp"
printf '%s\n' "--options-file \"${work}/cycle.rsp\"" > "${work}/cycle.rsp"
expect_error 'expects a file' --options-file
expect_error 'nonempty file names' --options-file=
expect_error 'nonempty file names' --options-file "${work}/empty.rsp,"
expect_error 'failed to open file' --options-file "${work}/missing.rsp"
expect_error 'unterminated quote or escape' --options-file "${work}/quote.rsp"
expect_error 'unterminated quote or escape' --options-file "${work}/escape.rsp"
expect_error 'nesting exceeds' --options-file "${work}/cycle.rsp"

echo 'PASS: nvcc options files preserve quoted includes and computed host values'
