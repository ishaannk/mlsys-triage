#!/bin/sh
# Runs ON THE POD. Deletes this pod after SD_SLEEP seconds, no matter what
# happens to the machine that launched it.
#
# The supervisor terminates pods in a `finally`, but a hard-killed process has
# no `finally`. A session ended mid-run once and left a pod billing until the
# next manual reap; at ~$0.49/hr an unnoticed orphan is ~$12/day. This is the
# backstop that does not depend on the laptop staying alive.
sleep "${SD_SLEEP:-7200}"
curl -s -X DELETE \
  -H "Authorization: Bearer ${SD_KEY}" \
  -H "User-Agent: runpod-python/1.7" \
  "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
  > /workspace/selfdestruct.log 2>&1
