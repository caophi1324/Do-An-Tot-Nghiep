@echo off
REM ============================================================
REM DRONEEXE - Start ArduPilot SITL Simulator
REM ============================================================
REM
REM Prerequisites:
REM   1. Python 3.10+ installed
REM   2. Install MAVProxy: pip install MAVProxy
REM   3. Download ArduPilot SITL:
REM      - Option A (WSL): Follow https://ardupilot.org/dev/docs/building-setup-linux.html
REM      - Option B (Native): Download pre-built from https://firmware.ardupilot.org/
REM
REM This script starts SITL with default settings:
REM   - Vehicle: ArduCopter
REM   - Location: Default (Canberra, Australia)
REM   - Port: 14550 (MAVLink UDP)
REM ============================================================

echo =============================================
echo   DRONEEXE - ArduPilot SITL Simulator
echo =============================================
echo.

REM Check if sim_vehicle.py is available
where sim_vehicle.py >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] sim_vehicle.py not found in PATH.
    echo.
    echo To install ArduPilot SITL on Windows:
    echo.
    echo Option 1 - WSL ^(Recommended^):
    echo   1. Install WSL: wsl --install
    echo   2. In WSL, run:
    echo      sudo apt-get update
    echo      sudo apt-get install git python3-pip
    echo      git clone https://github.com/ArduPilot/ardupilot.git
    echo      cd ardupilot
    echo      git submodule update --init --recursive
    echo      Tools/environment_install/install-prereqs-ubuntu.sh -y
    echo      . ~/.profile
    echo   3. Then run: sim_vehicle.py -v ArduCopter --map --console
    echo.
    echo Option 2 - Mission Planner SITL:
    echo   1. Download Mission Planner from https://firmware.ardupilot.org/Tools/MissionPlanner/
    echo   2. In Mission Planner: Simulation tab -^> select Copter
    echo   3. It will start SITL on localhost:14550
    echo.
    echo Option 3 - Docker:
    echo   docker run -it --rm -p 14550:14550/udp ardupilot/sitl-copter
    echo.
    pause
    exit /b 1
)

echo [*] Starting ArduPilot SITL (ArduCopter)...
echo [*] MAVLink will be available on udp:127.0.0.1:14550
echo [*] Press Ctrl+C to stop
echo.

sim_vehicle.py -v ArduCopter -L CMAC --map --console --out=udp:127.0.0.1:14550

pause
