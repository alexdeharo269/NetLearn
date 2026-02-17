$BASH = "C:\msys64\usr\bin\bash.exe"

# CAMBIO CLAVE: $PSScriptRoot nos lleva directo a la carpeta 'ESNcpp'
$PATH_WIN = $PSScriptRoot 

# Convertimos la ruta de Windows a formato MSYS2
$PATH_MSYS = & $BASH -lc "cd '$PATH_WIN'; pwd -W"

# Ejecutamos make. Como ya estamos dentro de la carpeta, el -f apunta solo al archivo.
$COMMAND = "export PATH=/mingw64/bin:`$PATH; cd '$PATH_MSYS' && make -f make.mak $($args -join ' ')"

& $BASH -lc $COMMAND