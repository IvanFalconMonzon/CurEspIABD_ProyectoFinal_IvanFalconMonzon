@echo off
REM --- Script de inicio para la aplicación de detección de hardware ---

REM Cambia al directorio del script (la raíz del proyecto)
cd /d "%~dp0"

REM --- Paso 1: Activar o crear el entorno virtual ---
echo.
echo =======================================================
echo  Activando el entorno virtual...
echo =======================================================

REM Verifica si el entorno virtual existe, si no, lo crea.
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo [INFO] Creando entorno virtual 'venv'...
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo ERROR: Fallo al crear el entorno virtual.
        echo Asegurate de tener Python instalado y en tu PATH.
        pause > NUL
        exit /b %ERRORLEVEL%
    )
)

REM Activa el entorno virtual. 'call' es importante para que el control regrese al script.
call venv\Scripts\activate.bat
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Fallo al activar el entorno virtual.
    echo Asegurate de que 'venv\Scripts\activate.bat' existe.
    pause > NUL
    exit /b %ERRORLEVEL%
)
echo Entorno virtual activado correctamente.

REM --- Paso 2: Instalar/Verificar dependencias ---
echo.
echo =======================================================
echo  Comprobando e instalando dependencias de Python...
echo =======================================================
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Fallo al instalar las dependencias de Python.
    echo Asegurate de tener acceso a internet.
    echo Presiona cualquier tecla para salir...
    pause > NUL
    exit /b %ERRORLEVEL%
)
echo.
echo Dependencias instaladas/verificadas correctamente.

REM --- Paso 3: Iniciar la aplicación Flask ---
echo.
echo ===========================================
echo  Iniciando la aplicacion Flask...
echo ===========================================

REM **CAMBIO CRUCIAL AQUÍ: Ejecuta directamente app.py con el Python del venv**
"%~dp0venv\Scripts\python.exe" app.py

REM El argumento --host, --port, --debug se pasan directamente al script app.py
REM si app.py está configurado para leerlos.
REM Para Flask, si no usas 'flask run', la forma más común de configurarlo es
REM directamente en app.py, por ejemplo:
REM if __name__ == '__main__':
REM     app.run(host='0.0.0.0', port=5000, debug=True)
REM Asegúrate de que tu app.py tiene esta última parte para que reciba esos parámetros.

echo.
echo La aplicacion Flask ha finalizado.
echo Presiona cualquier tecla para cerrar esta ventana...
pause > NUL