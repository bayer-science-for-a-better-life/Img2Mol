#!/usr/bin/env bash

gURL=https://drive.google.com/file/d/1pk21r4Zzb9ZJkszJwP9SObTlfTaRMMtF/view?usp=sharing
# match more than 26 word characters
ggID=$(echo "$gURL" | egrep -o '(\w|-){26,}')
# alternative, just hardcode the id
ggID='1pk21r4Zzb9ZJkszJwP9SObTlfTaRMMtF'
ggURL='https://drive.google.com/uc?export=download'

curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" >/dev/null
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"

FILE=/model/model.ckpt
if test -f "$FILE"; then
    echo "$FILE exists."
else
    echo "$FILE does not exist."
    echo -e "Downloading from "$gURL"...\n"
    cmd='curl --insecure -C - -LOJb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}"'
    eval $cmd
    mv 'model.ckpt' 'model/'
fi