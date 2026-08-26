#!/usr/bin/env bash

# Classify a manifest/observed-status transition without running a sample.
# `run-unverified` is intentionally provisional: once the sweep observes any
# concrete runtime result, the manifest must record that evidence.
cumetal_sweep_transition() {
    local expected="${1:?expected status required}"
    local actual="${2:?actual status required}"

    if [[ "${actual}" == "${expected}" ]]; then
        echo "match"
    elif [[ "${expected}" == "pass" || "${expected}" == "waive" ]]; then
        echo "regression"
    elif [[ "${expected}" == "run-unverified" ]]; then
        echo "evidence-update"
    elif [[ "${actual}" == "pass" || "${actual}" == "waive" ]]; then
        echo "improvement"
    else
        echo "unsupported-drift"
    fi
}
