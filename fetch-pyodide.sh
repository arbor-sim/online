#!/usr/bin/env bash

rm -fr static/pyodide
mkdir -p static

PYODIDE_VERSION=$(cat PYODIDE_VERSION)
PYODIDE_RELEASE_URL="https://github.com/pyodide/pyodide/releases/download"
PYODIDE_BASENAME=pyodide-${PYODIDE_VERSION}.tar.bz2

wget -nc "${PYODIDE_RELEASE_URL}/${PYODIDE_VERSION}/${PYODIDE_BASENAME}"
tar -xvf ${PYODIDE_BASENAME}
mv pyodide static

