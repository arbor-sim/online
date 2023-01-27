if [ ! -d 'static/pyodide' ]
then
    bash fetch-pyodide.sh
fi

echo "

Be sure to download the .whl file from the build
artifacts that matches the pyodide version, put
it in the static folder and then put the name of
that file in the ARBOR_WHEEL_NAME file in static

"

python3 \
    -m http.server \
    --directory static \
    4000

xdg-open localhost:4000
