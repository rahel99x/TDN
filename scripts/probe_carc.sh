#!/usr/bin/env bash
# Read-only: no allocation, install, numerical work, or configuration changes.
set -uo pipefail
printf 'Identity and scheduler\n'
hostname
id
command -v sbatch || true
printf '\nCurrent user account associations and quotas\n'
command -v myaccount >/dev/null && myaccount || true
command -v myquota >/dev/null && myquota || true
if command -v sacctmgr >/dev/null; then
    sacctmgr -nP show associations where user=aadaniel account=anakano_81 \
        format=Account,User,Partition,QOS,MaxWall,GrpTRES || true
fi
printf '\nLive partitions, GPUs, features and limits\n'
command -v sinfo >/dev/null && sinfo -h -o '%P|%a|%G|%l|%c|%m' || true
command -v sinfo >/dev/null && sinfo -p gpu -N -h -o '%N|%G|%f|%m|%c' || true
command -v scontrol >/dev/null && scontrol show partition || true
printf '\nModule inventory (Python and CUDA toolkits)\n'
if type module >/dev/null 2>&1; then
    module avail python 2>&1 || true
    module avail cuda 2>&1 || true
else
    printf 'Run from a CARC login shell to inspect the module inventory.\n'
fi
printf '\nRequests use user aadaniel, account anakano_81, and /home1/aadaniel/projects/TDN.\n'
printf 'Visible partitions do not prove account authorization; submission checks it live.\n'
