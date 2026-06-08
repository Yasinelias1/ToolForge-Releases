import os
os.environ['WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS'] = '--allow-file-access-from-files --disable-web-security --disable-http-cache --disk-cache-size=1 --media-cache-size=1'
import sys
import json
import base64
import tempfile
import subprocess
import threading
import requests
from io import BytesIO
import webview
import cv2
from PIL import Image
import qrcode
from static_ffmpeg import run
import numpy as np
import pypdf

# Path helper for PyInstaller resources
def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def get_executable_dir():
    if hasattr(sys, 'frozen'):
        return os.path.dirname(sys.executable)
    return os.path.abspath(".")

def get_iou(boxA, boxB):
    # box = (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0
    return interArea / unionArea

WMO_CODES = {
    0: {"en": "Clear sky", "de": "Klarer Himmel", "emoji": "☀️"},
    1: {"en": "Mainly clear", "de": "Überwiegend klar", "emoji": "🌤️"},
    2: {"en": "Partly cloudy", "de": "Teilweise bewölkt", "emoji": "⛅"},
    3: {"en": "Overcast", "de": "Bedeckt", "emoji": "☁️"},
    45: {"en": "Fog", "de": "Nebel", "emoji": "🌫️"},
    48: {"en": "Depositing rime fog", "de": "Raureifnebel", "emoji": "🌫️"},
    51: {"en": "Light drizzle", "de": "Leichter Nieselregen", "emoji": "🌧️"},
    53: {"en": "Moderate drizzle", "de": "Nieselregen", "emoji": "🌧️"},
    55: {"en": "Dense drizzle", "de": "Starker Nieselregen", "emoji": "🌧️"},
    56: {"en": "Light freezing drizzle", "de": "Leichter gefrierender Nieselregen", "emoji": "🌨️"},
    57: {"en": "Dense freezing drizzle", "de": "Starker gefrierender Nieselregen", "emoji": "🌨️"},
    61: {"en": "Slight rain", "de": "Leichter Regen", "emoji": "🌧️"},
    63: {"en": "Moderate rain", "de": "Regen", "emoji": "🌧️"},
    65: {"en": "Heavy rain", "de": "Starker Regen", "emoji": "🌧️"},
    66: {"en": "Light freezing rain", "de": "Leichter gefrierender Regen", "emoji": "🌨️"},
    67: {"en": "Heavy freezing rain", "de": "Starker gefrierender Regen", "emoji": "🌨️"},
    71: {"en": "Slight snow fall", "de": "Leichter Schneefall", "emoji": "❄️"},
    73: {"en": "Moderate snow fall", "de": "Schneefall", "emoji": "❄️"},
    75: {"en": "Heavy snow fall", "de": "Starker Schneefall", "emoji": "❄️"},
    77: {"en": "Snow grains", "de": "Schneegriesel", "emoji": "❄️"},
    80: {"en": "Slight rain showers", "de": "Leichte Regenschauer", "emoji": "🌧️"},
    81: {"en": "Moderate rain showers", "de": "Regenschauer", "emoji": "🌧️"},
    82: {"en": "Violent rain showers", "de": "Starke Regenschauer", "emoji": "🌧️"},
    85: {"en": "Slight snow showers", "de": "Leichte Schneeschauer", "emoji": "🌨️"},
    86: {"en": "Heavy snow showers", "de": "Starke Schneeschauer", "emoji": "🌨️"},
    95: {"en": "Thunderstorm", "de": "Gewitter", "emoji": "⛈️"},
    96: {"en": "Thunderstorm with slight hail", "de": "Gewitter mit leichtem Hagel", "emoji": "⛈️"},
    99: {"en": "Thunderstorm with heavy hail", "de": "Gewitter mit schwerem Hagel", "emoji": "⛈️"}
}

class ToolForgeAPI:
    APP_VERSION = "1.1.4"

    def __init__(self):
        self._window = None
        self._ffmpeg_path = None
        self._ffprobe_path = None
        self._init_ffmpeg()
        self._gpu_name = self._detect_gpu_name_once()
        
        # Caching variables for CPU and GPU details (updated asynchronously)
        self._cached_cpu_temp = None
        self._cached_gpu_pct = 0.0
        self._cached_gpu_temp = None
        self._cached_gpu_detected = False
        
        # Start background hardware stats polling thread
        self._hw_thread = threading.Thread(target=self._background_hw_polling, daemon=True)
        self._hw_thread.start()

    def _background_hw_polling(self):
        import shutil
        import time
        counter = 0
        while True:
            # 1. Query GPU stats (runs every 0.5 seconds, fast)
            gpu_pct = 0.0
            gpu_temp = None
            gpu_detected = False
            try:
                if shutil.which("nvidia-smi"):
                    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, startupinfo=startupinfo, timeout=1.0)
                    if res.returncode == 0 and res.stdout.strip():
                        parts = res.stdout.strip().split(",")
                        if len(parts) >= 2:
                            gpu_pct = float(parts[0].strip())
                            gpu_temp = int(parts[1].strip())
                            gpu_detected = True
            except:
                pass
                
            # 2. Query CPU stats (runs every 2.0 seconds, slow due to PowerShell overhead)
            # 0.5s * 4 = 2.0s
            if counter % 4 == 0 or self._cached_cpu_temp is None:
                cpu_temp = self._query_cpu_temp()
            else:
                cpu_temp = self._cached_cpu_temp
            
            # 3. Update cache
            self._cached_gpu_pct = gpu_pct
            self._cached_gpu_temp = gpu_temp
            self._cached_gpu_detected = gpu_detected
            self._cached_cpu_temp = cpu_temp
            
            counter = (counter + 1) % 1000
            
            # Sleep 0.5 seconds
            time.sleep(0.5)

    def _detect_gpu_name_once(self):
        try:
            import shutil
            if shutil.which("nvidia-smi"):
                cmd = ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, startupinfo=startupinfo, timeout=2)
                if res.returncode == 0 and res.stdout.strip():
                    return res.stdout.strip().split("\n")[0]
        except:
            pass
            
        try:
            cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"]
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, startupinfo=startupinfo, timeout=3)
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip().split("\n")[0]
        except:
            pass
            
        return "Standard-Grafikkarte"

    def _query_cpu_temp(self):
        try:
            cmd = ["powershell", "-NoProfile", "-Command", "Get-CimInstance -Namespace root/OpenHardwareMonitor -ClassName Sensor | Where-Object { $_.SensorType -eq 'Temperature' -and $_.Name -like '*CPU*' } | Select-Object -ExpandProperty Value"]
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, startupinfo=startupinfo, timeout=1.5)
            if res.returncode == 0 and res.stdout.strip():
                vals = [float(val) for val in res.stdout.strip().split("\n") if val.strip()]
                if vals:
                    return round(sum(vals) / len(vals))
        except:
            pass

        try:
            cmd = ["powershell", "-NoProfile", "-Command", "(Get-CimInstance -Namespace root/WMI -ClassName MSAcpi_ThermalZoneTemperature).CurrentTemperature"]
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            res = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, startupinfo=startupinfo, timeout=1.5)
            if res.returncode == 0 and res.stdout.strip():
                lines = res.stdout.strip().split("\n")
                vals = []
                for line in lines:
                    if line.strip():
                        temp_k = float(line.strip())
                        temp_c = (temp_k / 10.0) - 273.15
                        if 0 < temp_c < 110:
                            vals.append(temp_c)
                if vals:
                    return round(sum(vals) / len(vals))
        except:
            pass
        return None

    def get_system_stats(self):
        try:
            import psutil
            
            # Fast, in-thread psutil queries
            cpu_pct = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            ram_used_gb = ram.used / (1024 ** 3)
            ram_total_gb = ram.total / (1024 ** 3)
            
            # Subprocess-heavy values are read instantly from the background cache
            return {
                "success": True,
                "cpu_pct": cpu_pct,
                "ram_pct": ram.percent,
                "ram_used": f"{ram_used_gb:.1f}",
                "ram_total": f"{ram_total_gb:.1f}",
                "gpu_name": self._gpu_name,
                "gpu_pct": self._cached_gpu_pct if self._cached_gpu_detected else None,
                "gpu_temp": self._cached_gpu_temp,
                "cpu_temp": self._cached_cpu_temp
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_speed_test(self):
        try:
            import time
            import requests
            import threading
            import socket
            
            session = requests.Session()
            
            # Variables for the background updater thread
            self.speedtest_active = True
            self.speedtest_phase = 'ping'
            self.speedtest_pct = 0
            self.speedtest_value = 0.0

            def update_loop():
                last_phase = None
                last_pct = None
                last_val = None
                while self.speedtest_active:
                    phase = self.speedtest_phase
                    pct = self.speedtest_pct
                    val = self.speedtest_value
                    if phase != last_phase or pct != last_pct or val != last_val:
                        if phase == 'ping':
                            self._window.evaluate_js(f"if (typeof updateSpeedtestStatus === 'function') updateSpeedtestStatus('{phase}', {pct}, {val:.0f});")
                        else:
                            self._window.evaluate_js(f"if (typeof updateSpeedtestStatus === 'function') updateSpeedtestStatus('{phase}', {pct}, {val:.2f});")
                        last_phase = phase
                        last_pct = pct
                        last_val = val
                    time.sleep(0.15)

            updater_thread = threading.Thread(target=update_loop, daemon=True)
            updater_thread.start()

            # 1. Ping Phase
            pings = []
            
            # 1.1 Try UDP DNS query ping to 1.1.1.1:53 (extremely fast, connectionless, bypasses firewall SYN rate limiting)
            dns_query = b'\xaa\xbb\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x05speed\ncloudflare\x03com\x00\x00\x01\x00\x01'
            for i in range(5):
                if not self.speedtest_active:
                    break
                t0 = time.time()
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.settimeout(0.3)
                    s.sendto(dns_query, ("1.1.1.1", 53))
                    data, addr = s.recvfrom(512)
                    pings.append((time.time() - t0) * 1000)
                except:
                    pass
                finally:
                    s.close()
                self.speedtest_pct = int(((i + 1) / 5) * 100)
                self.speedtest_value = sum(pings) / len(pings) if pings else 0
                time.sleep(0.3)
                
            # 1.2 Fallback to TCP ping on Port 443 (HTTPS - never blocked, extremely reliable)
            if not pings and self.speedtest_active:
                for i in range(5):
                    if not self.speedtest_active:
                        break
                    t0 = time.time()
                    try:
                        s = socket.create_connection(("speed.cloudflare.com", 443), timeout=0.4)
                        s.close()
                        pings.append((time.time() - t0) * 1000)
                    except:
                        pass
                    self.speedtest_pct = int(((i + 1) / 5) * 100)
                    self.speedtest_value = sum(pings) / len(pings) if pings else 0
                    time.sleep(0.3)

            # 1.3 Fallback to TCP ping on Port 80 (HTTP) if Port 443 also failed
            if not pings and self.speedtest_active:
                for i in range(5):
                    if not self.speedtest_active:
                        break
                    t0 = time.time()
                    try:
                        s = socket.create_connection(("speed.cloudflare.com", 80), timeout=0.4)
                        s.close()
                        pings.append((time.time() - t0) * 1000)
                    except:
                        pass
                    self.speedtest_pct = int(((i + 1) / 5) * 100)
                    self.speedtest_value = sum(pings) / len(pings) if pings else 0
                    time.sleep(0.3)
                    
            # 1.4 Fallback to HTTP ping if all TCP failed
            if not pings and self.speedtest_active:
                try:
                    session.get("https://speed.cloudflare.com/__down?bytes=0", timeout=1.0)
                except:
                    pass
                for i in range(5):
                    if not self.speedtest_active:
                        break
                    t0 = time.time()
                    try:
                        res = session.get("https://speed.cloudflare.com/__down?bytes=0", timeout=1.0)
                        res.raise_for_status()
                        pings.append((time.time() - t0) * 1000)
                    except:
                        pass
                    self.speedtest_pct = int(((i + 1) / 5) * 100)
                    self.speedtest_value = sum(pings) / len(pings) if pings else 0
                    time.sleep(0.3)
                    
            ping = sum(pings) / len(pings) if pings else 20.0

            
            # 2. Download Phase
            self.speedtest_phase = 'download'
            self.speedtest_pct = 0
            self.speedtest_value = 0.0
            
            dl_start = time.time()
            downloaded = 0
            
            stable_dl_start = None
            stable_downloaded = 0
            
            while self.speedtest_active:
                elapsed = time.time() - dl_start
                if elapsed >= 5.0:
                    break
                
                # Cloudflare supports up to 50MB (50000000 bytes)
                dl_url = "https://speed.cloudflare.com/__down?bytes=50000000"
                try:
                    with session.get(dl_url, stream=True, timeout=5) as r:
                        r.raise_for_status()
                        for chunk in r.iter_content(chunk_size=128 * 1024): # 128KB chunks
                            if not self.speedtest_active:
                                break
                            if chunk:
                                downloaded += len(chunk)
                                el = time.time() - dl_start
                                if el >= 5.0:
                                    break
                                
                                # Stable average calculation after 1.5 seconds
                                if el >= 1.5:
                                    if stable_dl_start is None:
                                        stable_dl_start = time.time()
                                        stable_downloaded = len(chunk)
                                    else:
                                        stable_downloaded += len(chunk)
                                
                                # Calculate current speed
                                if stable_dl_start is not None:
                                    stable_elapsed = time.time() - stable_dl_start
                                    current_speed = (stable_downloaded * 8) / (stable_elapsed * 1000000) if stable_elapsed > 0 else 0.0
                                else:
                                    current_speed = (downloaded * 8) / (el * 1000000) if el >= 0.2 else 0.0
                                
                                self.speedtest_pct = min(99, int((el / 5.0) * 100))
                                self.speedtest_value = current_speed
                except:
                    pass
            
            dl_duration = time.time() - dl_start
            if stable_dl_start is not None:
                stable_elapsed = time.time() - stable_dl_start
                final_dl_speed = (stable_downloaded * 8) / (stable_elapsed * 1000000) if stable_elapsed > 0 else 0.0
            else:
                final_dl_speed = (downloaded * 8) / (dl_duration * 1000000) if dl_duration >= 0.5 else 0.0
            
            # 3. Upload Phase
            self.speedtest_phase = 'upload'
            self.speedtest_pct = 0
            self.speedtest_value = 0.0
            
            ul_url = "https://speed.cloudflare.com/__up"
            ul_start = time.time()
            uploaded = 0
            chunk_data = os.urandom(128 * 1024) # 128KB chunks
            
            stable_ul_start = None
            stable_uploaded = 0
            
            def upload_generator():
                nonlocal uploaded, stable_ul_start, stable_uploaded
                while self.speedtest_active:
                    elapsed = time.time() - ul_start
                    if elapsed >= 5.0:
                        break
                    yield chunk_data
                    uploaded += len(chunk_data)
                    
                    # Stable average calculation after 1.5 seconds
                    if elapsed >= 1.5:
                        if stable_ul_start is None:
                            stable_ul_start = time.time()
                            stable_uploaded = len(chunk_data)
                        else:
                            stable_uploaded += len(chunk_data)
                    
                    # Calculate current speed
                    if stable_ul_start is not None:
                        stable_elapsed = time.time() - stable_ul_start
                        current_speed = (stable_uploaded * 8) / (stable_elapsed * 1000000) if stable_elapsed > 0 else 0.0
                    else:
                        current_speed = (uploaded * 8) / (elapsed * 1000000) if elapsed >= 0.2 else 0.0
                    
                    self.speedtest_pct = min(99, int((elapsed / 5.0) * 100))
                    self.speedtest_value = current_speed
            
            try:
                r_up = session.post(ul_url, data=upload_generator(), timeout=10)
                r_up.raise_for_status()
            except:
                pass
            
            ul_duration = time.time() - ul_start
            if stable_ul_start is not None:
                stable_elapsed = time.time() - stable_ul_start
                final_ul_speed = (stable_uploaded * 8) / (stable_elapsed * 1000000) if stable_elapsed > 0 else 0.0
            else:
                final_ul_speed = (uploaded * 8) / (ul_duration * 1000000) if ul_duration > 0 else 0.0
            
            # Stop updater thread
            self.speedtest_active = False
            updater_thread.join(timeout=0.5)
            
            # 4. Final Updates to frontend (outside background thread, direct and synchronous to set exact final state)
            self._window.evaluate_js(f"if (typeof updateSpeedtestStatus === 'function') updateSpeedtestStatus('download', 100, {final_dl_speed:.2f});")
            self._window.evaluate_js(f"if (typeof updateSpeedtestStatus === 'function') updateSpeedtestStatus('upload', 100, {final_ul_speed:.2f});")
            self._window.evaluate_js(f"if (typeof updateSpeedtestStatus === 'function') updateSpeedtestStatus('complete', 100, {final_ul_speed:.2f});")
            
            return {
                "success": True,
                "download": f"{final_dl_speed:.2f}",
                "upload": f"{final_ul_speed:.2f}",
                "ping": f"{ping:.0f}"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_config_path(self):
        return os.path.join(get_executable_dir(), "config.json")

    def _load_config(self):
        config_path = self._get_config_path()
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except:
                pass
        return {
            "github_repo": "Yasinelias1/ToolForge-Releases",
            "current_version": self.APP_VERSION
        }

    def _save_config(self, config):
        try:
            config_path = self._get_config_path()
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)
            return True
        except Exception as e:
            print("Error saving config:", e)
            return False

    def get_app_version(self):
        return self.APP_VERSION

    def get_github_repo(self):
        config = self._load_config()
        return config.get("github_repo", "Yasinelias1/ToolForge-Releases")

    def save_github_repo(self, repo):
        config = self._load_config()
        config["github_repo"] = repo.strip()
        self._save_config(config)
        return {"success": True}

    def get_language(self):
        config = self._load_config()
        return config.get("language", "de")

    def save_language(self, lang):
        try:
            config = self._load_config()
            config["language"] = lang.strip().lower()
            self._save_config(config)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_theme(self):
        try:
            config = self._load_config()
            return config.get("theme", "sunset")
        except:
            return "sunset"

    def save_theme(self, theme):
        try:
            config = self._load_config()
            config["theme"] = theme.strip().lower()
            self._save_config(config)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_dashboard_options(self):
        try:
            config = self._load_config()
            return config.get("dashboard_options", {
                "optCpu": True,
                "optRam": True,
                "optGpu": True,
                "optChart": True,
                "optWeather": True
            })
        except:
            return {
                "optCpu": True,
                "optRam": True,
                "optGpu": True,
                "optChart": True,
                "optWeather": True
            }

    def save_dashboard_options(self, options):
        try:
            config = self._load_config()
            if isinstance(options, str):
                import json
                opts = json.loads(options)
            else:
                opts = options
            config["dashboard_options"] = opts
            self._save_config(config)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def reset_all_settings(self):
        try:
            config = {
                "github_repo": "Yasinelias1/ToolForge-Releases",
                "current_version": self.APP_VERSION,
                "language": "de",
                "theme": "sunset",
                "dashboard_options": {
                    "optCpu": True,
                    "optRam": True,
                    "optGpu": True,
                    "optChart": True,
                    "optWeather": True
                }
            }
            self._save_config(config)
            return {"success": True, "defaults": config}
        except Exception as e:
            return {"success": False, "error": str(e)}



    def check_for_updates(self):
        try:
            import re
            repo = self.get_github_repo()
            if not repo or "/" not in repo:
                return {"success": False, "error": "Ungültiges GitHub-Repository Format (muss 'Benutzername/Projekt' sein)."}
            
            url = f"https://api.github.com/repos/{repo}/releases/latest"
            headers = {"User-Agent": "ToolForge-Updater"}
            
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code == 404:
                return {"success": False, "error": f"Repository '{repo}' oder keine Releases gefunden."}
            elif r.status_code != 200:
                return {"success": False, "error": f"GitHub API Fehler: Status {r.status_code}"}
                
            data = r.json()
            latest_tag = data.get("tag_name", "0.0.0")
            latest_version = latest_tag.lower().lstrip("v")
            current_version = self.APP_VERSION.lower().lstrip("v")
            
            def parse_ver(v_str):
                try:
                    return [int(x) for x in re.sub(r'[^\d.]', '', v_str).split(".")]
                except:
                    return [0, 0, 0]
            
            curr_parts = parse_ver(current_version)
            late_parts = parse_ver(latest_version)
            
            while len(curr_parts) < len(late_parts):
                curr_parts.append(0)
            while len(late_parts) < len(curr_parts):
                late_parts.append(0)
                
            update_available = late_parts > curr_parts
            
            download_url = None
            assets = data.get("assets", [])
            for asset in assets:
                name = asset.get("name", "").lower()
                if name.endswith(".zip"):
                    download_url = asset.get("browser_download_url")
                    break
            
            if update_available and not download_url:
                download_url = data.get("zipball_url")
                
            return {
                "success": True,
                "update_available": update_available,
                "latest_version": latest_tag,
                "changelog": data.get("body", "Keine Beschreibung verfügbar."),
                "download_url": download_url
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def install_update(self, download_url):
        try:
            if not hasattr(sys, 'frozen'):
                return {"success": False, "error": "Updates können nur in der fertig gebauten EXE-Version installiert werden. (Entwicklungsmodus aktiv)"}
            
            if not download_url:
                return {"success": False, "error": "Keine Download-URL vorhanden."}
            
            import zipfile
            import shutil
            
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "update.zip")
            
            # Download file
            headers = {"User-Agent": "ToolForge-Updater"}
            r = requests.get(download_url, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract
            extract_path = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Find the directory containing ToolForge.exe
            src_dir = extract_path
            for root, dirs, files in os.walk(extract_path):
                if "ToolForge.exe" in files:
                    src_dir = root
                    break
            
            exe_dir = get_executable_dir()
            batch_path = os.path.join(temp_dir, "install_update.bat")
            
            # Write batch file
            batch_content = f"""@echo off
chcp 65001 > nul
echo Installiere ToolForge Update...
echo Bitte warten, bis das Update abgeschlossen ist.
timeout /t 2 /nobreak > nul

taskkill /f /im ToolForge.exe > nul 2>&1
timeout /t 1 /nobreak > nul

xcopy /y /s /e "{src_dir}\\*.*" "{exe_dir}\\" > nul

echo Update erfolgreich! Starte ToolForge neu...
start "" "{exe_dir}\\ToolForge.exe"

del "%~f0"
"""
            with open(batch_path, "w", encoding="utf-8") as f:
                f.write(batch_content)
            
            # Popen cmd in background
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.Popen(f'cmd.exe /c "{batch_path}"', shell=True, startupinfo=startupinfo)
            
            # Close window and exit immediately so batch file can write files without block
            if self._window:
                self._window.destroy()
            sys.exit(0)
            
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _set_window(self, window):
        self._window = window

    def _init_ffmpeg(self):
        try:
            if hasattr(sys, '_MEIPASS'):
                # In PyInstaller, we bundle static-ffmpeg bin into sys._MEIPASS/static_ffmpeg/bin
                ffmpeg_exe = os.path.join(sys._MEIPASS, 'static_ffmpeg', 'bin', 'win32', 'ffmpeg.exe')
                ffprobe_exe = os.path.join(sys._MEIPASS, 'static_ffmpeg', 'bin', 'win32', 'ffprobe.exe')
                if os.path.exists(ffmpeg_exe):
                    self._ffmpeg_path = ffmpeg_exe
                    self._ffprobe_path = ffprobe_exe
                    print("Initialized bundled ffmpeg:", self._ffmpeg_path)
                    return
            
            # static-ffmpeg provides the paths to platform-specific binaries
            ffmpeg_exe, ffprobe_exe = run.get_or_fetch_platform_executables_else_raise()
            self._ffmpeg_path = ffmpeg_exe
            self._ffprobe_path = ffprobe_exe
            print("Initialized development ffmpeg:", self._ffmpeg_path)
        except Exception as e:
            print("Failed to initialize ffmpeg:", e)

    # ── Dialogs ──
    def select_file(self, file_types_json):
        try:
            if not self._window:
                return None
            raw_types = json.loads(file_types_json)
            file_types = []
            import re
            for item in raw_types:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    desc, mask = item
                    # pywebview's filter regex strictly requires only alphanumeric and spaces in the description.
                    # Strip any parentheses, commas, hyphens, etc.
                    clean_desc = re.sub(r'[^\w ]', ' ', desc)
                    # Normalize multiple spaces
                    clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                    file_types.append(f"{clean_desc} ({mask})")
                elif isinstance(item, str):
                    file_types.append(item)
            
            res = self._window.create_file_dialog(webview.OPEN_DIALOG, file_types=tuple(file_types))
            return res[0] if res else None
        except Exception as e:
            print("Error in select_file:", e)
            return None

    def select_folder(self):
        try:
            if not self._window:
                return {"success": False, "error": "Fenster-Instanz nicht bereit."}
            res = self._window.create_file_dialog(webview.FOLDER_DIALOG)
            return res[0] if res else None
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_file_dialog(self, file_types_json, default_name=""):
        try:
            if not self._window:
                return None
            raw_types = json.loads(file_types_json)
            file_types = []
            import re
            for item in raw_types:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    desc, mask = item
                    clean_desc = re.sub(r'[^\w ]', ' ', desc)
                    clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                    file_types.append(f"{clean_desc} ({mask})")
                elif isinstance(item, str):
                    file_types.append(item)
            
            res = self._window.create_file_dialog(webview.SAVE_DIALOG, file_types=tuple(file_types), save_filename=default_name)
            return res[0] if res else None
        except Exception as e:
            print("Error in save_file_dialog:", e)
            return None

    # ── Image Operations ──
    def get_file_size(self, path):
        try:
            if os.path.exists(path):
                return {"success": True, "size": os.path.getsize(path)}
            return {"success": False, "error": "Datei existiert nicht."}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def resize_image(self, image_path, width, height, format_type, quality, output_path):
        try:
            if isinstance(image_path, (list, tuple)):
                image_path = image_path[0]
            if isinstance(output_path, (list, tuple)):
                output_path = output_path[0]
            width = int(width)
            height = int(height)
            quality = int(quality)
            
            img = Image.open(image_path)
            # Convert RGBA to RGB if saving as JPEG
            if format_type.upper() in ["JPG", "JPEG"] and img.mode in ("RGBA", "LA", "P"):
                # create background white
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
                img = bg
            
            resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            pil_format = "JPEG" if format_type.upper() in ["JPG", "JPEG"] else format_type.upper()
            
            save_args = {}
            if pil_format in ["JPEG", "WEBP"]:
                save_args["quality"] = quality
                
            resized_img.save(output_path, format=pil_format, **save_args)
            return {"success": True, "size": os.path.getsize(output_path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def save_base64_image(self, base64_str, output_path):
        if isinstance(output_path, (list, tuple)):
            output_path = output_path[0]
        try:
            # base64_str format: "data:image/png;base64,iVBORw0KG..."
            header, encoded = base64_str.split(",", 1)
            data = base64.b64decode(encoded)
            with open(output_path, "wb") as f:
                f.write(data)
            return {"success": True, "size": len(data)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def inpaint_image(self, img_base64, mask_base64):
        try:
            # Decode main image
            header, encoded = img_base64.split(",", 1)
            img_data = base64.b64decode(encoded)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return {"success": False, "error": "Konnte Hauptbild nicht dekodieren."}

            # Decode mask image
            header, encoded = mask_base64.split(",", 1)
            mask_data = base64.b64decode(encoded)
            nparr_mask = np.frombuffer(mask_data, np.uint8)
            mask_img = cv2.imdecode(nparr_mask, cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                return {"success": False, "error": "Konnte Maske nicht dekodieren."}

            h, w = img.shape[:2]
            # Resize mask if dimensions don't match exactly
            if mask_img.shape[0] != h or mask_img.shape[1] != w:
                mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

            # Create binary mask (single channel, uint8)
            if len(mask_img.shape) == 3 and mask_img.shape[2] == 4:
                # If transparent PNG, mask is the alpha channel
                mask = mask_img[:, :, 3]
            elif len(mask_img.shape) == 3:
                # If 3 channel BGR, convert to gray and threshold
                gray_mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray_mask, 10, 255, cv2.THRESH_BINARY)
            else:
                # Already single channel
                _, mask = cv2.threshold(mask_img, 10, 255, cv2.THRESH_BINARY)

            # Dilate the mask slightly (3x3 kernel) to ensure edge pixels of watermark are covered
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask = cv2.dilate(mask, kernel, iterations=1)

            # Inpaint using Telea
            dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

            # Encode result back to PNG base64
            success, buffer = cv2.imencode('.png', dst)
            if not success:
                return {"success": False, "error": "Konnte Ergebnisbild nicht enkodieren."}

            res_b64 = base64.b64encode(buffer).decode('utf-8')
            return {"success": True, "image": f"data:image/png;base64,{res_b64}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Video Operations & Frame Extract ──
    def get_video_frame(self, video_path):
        try:
            if isinstance(video_path, (list, tuple)):
                video_path = video_path[0]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"success": False, "error": "Konnte Video nicht öffnen."}
            
            # Read first frame or skip a bit to get a non-black frame (e.g. frame at 1 sec)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                fps = 25.0
            
            duration = total_frames / fps if fps > 0 else 0.0
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 0.5)) # 0.5 seconds in
            success, frame = cap.read()
            if not success:
                # Fallback to first frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()
                
            cap.release()
            
            if not success:
                return {"success": False, "error": "Konnte keinen Frame extrahieren."}
            
            # Downscale frame slightly if it's huge, to keep the UI fast
            h, w = frame.shape[:2]
            max_size = 1200
            if w > max_size or h > max_size:
                scale = max_size / max(w, h)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
            # Encode frame to base64 PNG
            success, buffer = cv2.imencode('.png', frame)
            if not success:
                return {"success": False, "error": "Konnte Bild nicht encodieren."}
            
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return {
                "success": True, 
                "frame": f"data:image/png;base64,{img_base64}",
                "original_width": w,
                "original_height": h,
                "duration": duration
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def process_video(self, video_path, rect_json, action, tracking=False, auto_face=False, start_time=-1.0, end_time=-1.0, autodetect=False, fill_color_hex="#000000"):
        temp_avi = None
        temp_mp4 = None
        try:
            tracking = bool(tracking)
            auto_face = bool(auto_face)
            autodetect = bool(autodetect)
            
            # Convert hex to BGR
            hex_clean = fill_color_hex.lstrip('#')
            if len(hex_clean) == 6:
                r = int(hex_clean[0:2], 16)
                g = int(hex_clean[2:4], 16)
                b = int(hex_clean[4:6], 16)
            else:
                r, g, b = 0, 0, 0
            fill_color = (b, g, r)
            
            if isinstance(video_path, (list, tuple)):
                video_path = video_path[0]

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"success": False, "error": "Konnte Video nicht öffnen."}
            
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0: fps = 25.0
            if total_frames <= 0: total_frames = 1

            # Setup detection or tracking parameters
            rx, ry, rw, rh = 0, 0, 0, 0
            if not auto_face:
                # Parse bounding box if we are in manual mode
                rect = json.loads(rect_json)
                rx = int(rect['x'])
                ry = int(rect['y'])
                rw = int(rect['w'])
                rh = int(rect['h'])
                # Constrain crop coords to valid bounds
                rx = max(0, min(orig_w - 1, rx))
                ry = max(0, min(orig_h - 1, ry))
                rw = max(1, min(orig_w - rx, rw))
                rh = max(1, min(orig_h - ry, rh))
            else:
                # Initialize Haar Cascade face detector
                cascade_path = get_resource_path('haarcascade_frontalface_alt2.xml')
                face_cascade = cv2.CascadeClassifier(cascade_path)
                tracked_faces = []
                lost_limit = 15  # Keep face box for up to 15 frames if lost

            # Create a temp output video path (AVI format using standard MJPG or MP4V)
            fd, temp_avi = tempfile.mkstemp(suffix='.avi')
            os.close(fd)

            # Create a temp output MP4 path inside the gui/tools/ folder
            import time
            gui_tools_dir = get_resource_path('gui/tools')
            temp_name = f'temp_video_{int(time.time())}.mp4'
            temp_mp4 = os.path.join(gui_tools_dir, temp_name)

            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(temp_avi, fourcc, fps, (orig_w, orig_h))
            
            if not writer.isOpened():
                cap.release()
                return {"success": False, "error": "Konnte VideoWriter nicht initialisieren."}

            tracker = None
            frame_idx = 0
            
            # Autodetect template variables
            template_gray = None
            
            while True:
                success, frame = cap.read()
                if not success:
                    break
                
                # Check manual time range
                current_time = frame_idx / fps
                in_time_range = True
                if start_time >= 0.0 and end_time >= 0.0:
                    in_time_range = (current_time >= start_time and current_time <= end_time)
                
                if not in_time_range:
                    # Write frame directly outside of range
                    writer.write(frame)
                    frame_idx += 1
                    if frame_idx % 10 == 0 or frame_idx == total_frames:
                        pct = int((frame_idx / total_frames) * 100)
                        self._window.evaluate_js(f"if (typeof updateVideoProgress === 'function') updateVideoProgress({pct});")
                    continue

                # Gather list of face/object boxes to censor in this frame
                faces_list = []
                is_visible = True

                if auto_face:
                    # Convert to grayscale for Haar Cascade
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    
                    # Optimize performance: Downscale frame for faster face detection
                    detect_w = 640
                    h, w = frame.shape[:2]
                    scale = w / detect_w
                    if scale > 1.0:
                        detect_h = int(h / scale)
                        small_gray = cv2.resize(gray, (detect_w, detect_h), interpolation=cv2.INTER_AREA)
                    else:
                        small_gray = gray
                        scale = 1.0

                    detected_faces_small = face_cascade.detectMultiScale(
                        small_gray, 
                        scaleFactor=1.1, 
                        minNeighbors=4, 
                        minSize=(int(30 / scale), int(30 / scale)),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    # Map coordinates back to original frame size
                    detected_faces = []
                    for (fx, fy, fw, fh) in detected_faces_small:
                        fx_orig = int(fx * scale)
                        fy_orig = int(fy * scale)
                        fw_orig = int(fw * scale)
                        fh_orig = int(fh * scale)
                        
                        fx_orig = max(0, min(orig_w - 1, fx_orig))
                        fy_orig = max(0, min(orig_h - 1, fy_orig))
                        fw_orig = max(1, min(orig_w - fx_orig, fw_orig))
                        fh_orig = max(1, min(orig_h - fy_orig, fh_orig))
                        detected_faces.append((fx_orig, fy_orig, fw_orig, fh_orig))
                    
                    new_tracked_faces = []
                    matched_detected_indices = set()
                    
                    for tf in tracked_faces:
                        best_iou = 0
                        best_idx = -1
                        for idx, db in enumerate(detected_faces):
                            if idx in matched_detected_indices:
                                continue
                            iou = get_iou(tf['box'], db)
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = idx
                                
                        if best_idx != -1 and best_iou >= 0.15:
                            matched_detected_indices.add(best_idx)
                            db = detected_faces[best_idx]
                            
                            alpha = 0.5
                            old_x, old_y, old_w, old_h = tf['box']
                            new_x = int(alpha * db[0] + (1 - alpha) * old_x)
                            new_y = int(alpha * db[1] + (1 - alpha) * old_y)
                            new_w = int(alpha * db[2] + (1 - alpha) * old_w)
                            new_h = int(alpha * db[3] + (1 - alpha) * old_h)
                            
                            new_tracked_faces.append({
                                'box': (new_x, new_y, new_w, new_h),
                                'lost_count': 0,
                                'detect_count': tf['detect_count'] + 1
                            })
                        else:
                            lost_cnt = tf['lost_count'] + 1
                            if lost_cnt <= lost_limit:
                                new_tracked_faces.append({
                                    'box': tf['box'],
                                    'lost_count': lost_cnt,
                                    'detect_count': tf['detect_count']
                                })
                                
                    for idx, db in enumerate(detected_faces):
                        if idx not in matched_detected_indices:
                            new_tracked_faces.append({
                                'box': (db[0], db[1], db[2], db[3]),
                                'lost_count': 0,
                                'detect_count': 1
                            })
                            
                    tracked_faces = new_tracked_faces
                    
                    for tf in tracked_faces:
                        fx, fy, fw, fh = tf['box']
                        fx = max(0, min(orig_w - 1, fx))
                        fy = max(0, min(orig_h - 1, fy))
                        fw = max(1, min(orig_w - fx, fw))
                        fh = max(1, min(orig_h - fy, fh))
                        faces_list.append((fx, fy, fw, fh))
                else:
                    # Update tracker if manual tracking is enabled
                    if tracking:
                        if frame_idx == 0 or (tracker is None and in_time_range):
                            tracker = cv2.TrackerMIL_create()
                            tracker.init(frame, (rx, ry, rw, rh))
                        else:
                            ok, bbox = tracker.update(frame)
                            if ok:
                                rx, ry, rw, rh = [int(v) for v in bbox]
                                rx = max(0, min(orig_w - 1, rx))
                                ry = max(0, min(orig_h - 1, ry))
                                rw = max(1, min(orig_w - rx, rw))
                                rh = max(1, min(orig_h - ry, rh))
                    
                    # Autodetect Template Matching
                    if autodetect:
                        roi = frame[ry:ry+rh, rx:rx+rw]
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        
                        if template_gray is None:
                            template_gray = roi_gray.copy()
                        
                        if roi_gray.shape == template_gray.shape:
                            match_res = cv2.matchTemplate(roi_gray, template_gray, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(match_res)
                            if max_val < 0.65:
                                is_visible = False
                        else:
                            is_visible = False
                    
                    if is_visible:
                        faces_list.append((rx, ry, rw, rh))

                # Apply zensur action to all boxes in faces_list
                if is_visible and faces_list:
                    if action == 'inpaint':
                        import numpy as np
                        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        for (fx, fy, fw, fh) in faces_list:
                            pad = 8
                            px = max(0, fx - pad)
                            py = max(0, fy - pad)
                            pw = min(orig_w - px, fw + 2 * pad)
                            ph = min(orig_h - py, fh + 2 * pad)
                            mask[py:py+ph, px:px+pw] = 255
                        frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
                    else:
                        for (fx, fy, fw, fh) in faces_list:
                            if action == 'black':
                                frame[fy:fy+fh, fx:fx+fw] = (0, 0, 0)
                            elif action == 'color':
                                frame[fy:fy+fh, fx:fx+fw] = fill_color
                            elif action == 'blur':
                                crop = frame[fy:fy+fh, fx:fx+fw]
                                ksize_w = int(fw / 1.5) // 2 * 2 + 1
                                ksize_h = int(fh / 1.5) // 2 * 2 + 1
                                ksize = min(199, max(31, ksize_w, ksize_h))
                                sigma = ksize / 3.0
                                blurred = cv2.GaussianBlur(crop, (ksize, ksize), sigma)
                                frame[fy:fy+fh, fx:fx+fw] = blurred
                            elif action == 'pixelate':
                                crop = frame[fy:fy+fh, fx:fx+fw]
                                pix_w = max(3, fw // 25)
                                pix_h = max(3, fh // 25)
                                small = cv2.resize(crop, (pix_w, pix_h), interpolation=cv2.INTER_LINEAR)
                                pixelated = cv2.resize(small, (fw, fh), interpolation=cv2.INTER_NEAREST)
                                frame[fy:fy+fh, fx:fx+fw] = pixelated
                
                writer.write(frame)
                frame_idx += 1
                
                if frame_idx % 10 == 0 or frame_idx == total_frames:
                    pct = int((frame_idx / total_frames) * 100)
                    self._window.evaluate_js(f"if (typeof updateVideoProgress === 'function') updateVideoProgress({pct});")

            cap.release()
            writer.release()

            # Mux audio using ffmpeg if we have it
            if self._ffmpeg_path and os.path.exists(self._ffmpeg_path):
                # Copy audio from original, video from processed, compress as H264 MP4 with ultrafast preset
                cmd = [
                    self._ffmpeg_path,
                    '-i', video_path,
                    '-i', temp_avi,
                    '-map', '1:v',
                    '-map', '0:a?',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-preset', 'ultrafast',
                    '-c:a', 'aac',
                    temp_mp4,
                    '-y'
                ]
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                subprocess.run(cmd, startupinfo=startupinfo, check=True)
            else:
                os.replace(temp_avi, temp_mp4)

            return {"success": True, "temp_path": temp_mp4}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            if temp_avi and os.path.exists(temp_avi):
                try: os.remove(temp_avi)
                except: pass

    def copy_file(self, src_path, dest_path):
        try:
            if isinstance(src_path, (list, tuple)):
                src_path = src_path[0]
            if isinstance(dest_path, (list, tuple)):
                dest_path = dest_path[0]
            import shutil
            shutil.copy2(src_path, dest_path)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def rename_file(self, src_path, dest_path):
        try:
            if isinstance(src_path, (list, tuple)): src_path = src_path[0]
            if isinstance(dest_path, (list, tuple)): dest_path = dest_path[0]
            
            if not os.path.exists(src_path):
                return {"success": False, "error": "Quelldatei existiert nicht."}
            
            dest_dir = os.path.dirname(dest_path)
            if not os.path.exists(dest_dir):
                return {"success": False, "error": "Zielordner existiert nicht."}
                
            os.rename(src_path, dest_path)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def prepare_video_preview(self, video_path):
        try:
            if isinstance(video_path, (list, tuple)):
                video_path = video_path[0]
            import shutil
            import time
            gui_tools_dir = get_resource_path('gui/tools')
            temp_name = f"temp_preview_{int(time.time())}.mp4"
            dest_path = os.path.join(gui_tools_dir, temp_name)
            shutil.copy2(video_path, dest_path)
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            return {
                "success": True, 
                "temp_name": temp_name,
                "fps": fps if fps > 0 else 25.0,
                "total_frames": total_frames
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_image_base64(self, image_path):
        try:
            if isinstance(image_path, (list, tuple)):
                image_path = image_path[0]
            with open(image_path, 'rb') as f:
                data = f.read()
            import base64
            _, ext = os.path.splitext(image_path)
            ext = ext.lower().lstrip('.')
            if ext == 'jpg': ext = 'jpeg'
            mime = f"image/{ext}"
            if ext not in ['png', 'jpeg', 'jpg', 'webp', 'gif']:
                mime = "image/png"
            img_b64 = base64.b64encode(data).decode('utf-8')
            return {"success": True, "base64": f"data:{mime};base64,{img_b64}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── GIF Creator ──
    def create_gif(self, video_path, start_time, duration, fps, width, output_path):
        try:
            if isinstance(video_path, (list, tuple)):
                video_path = video_path[0]
            if isinstance(output_path, (list, tuple)):
                output_path = output_path[0]
            start_time = float(start_time)
            duration = float(duration)
            fps = int(fps)
            width = int(width)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"success": False, "error": "Konnte Video nicht öffnen."}

            orig_fps = cap.get(cv2.CAP_PROP_FPS)
            if orig_fps <= 0: orig_fps = 25.0
            
            # Seek to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * orig_fps))
            
            # Frame collection
            frames_to_read = int(duration * fps)
            step = max(1.0, orig_fps / fps)
            
            collected_frames = []
            frame_idx = 0
            
            for i in range(frames_to_read):
                target_frame = int(start_time * orig_fps + i * step)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                success, frame = cap.read()
                if not success:
                    break
                
                # Resize frame
                h, w = frame.shape[:2]
                target_h = int(h * (width / w))
                resized = cv2.resize(frame, (width, target_h), interpolation=cv2.INTER_AREA)
                
                # Convert BGR (OpenCV) to RGB (Pillow)
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                collected_frames.append(pil_img)
                
                # Report progress
                pct = int(((i + 1) / frames_to_read) * 90) # up to 90%
                self._window.evaluate_js(f"if (typeof updateGifProgress === 'function') updateGifProgress({pct});")

            cap.release()
            
            if not collected_frames:
                return {"success": False, "error": "Keine Frames für das GIF extrahiert."}

            # Save as GIF
            duration_per_frame = int(1000 / fps)
            collected_frames[0].save(
                output_path,
                save_all=True,
                append_images=collected_frames[1:],
                duration=duration_per_frame,
                loop=0,
                optimize=True
            )
            
            self._window.evaluate_js(f"if (typeof updateGifProgress === 'function') updateGifProgress(100);")
            # Read saved GIF and convert to base64
            with open(output_path, 'rb') as f:
                gif_data = f.read()
            import base64
            gif_b64 = base64.b64encode(gif_data).decode('utf-8')
            return {
                "success": True,
                "size": len(gif_data),
                "base64": f"data:image/gif;base64,{gif_b64}"
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── QR Code Generator ──
    def generate_qr_code(self, text, fill_color, back_color, size, output_path):
        try:
            if isinstance(output_path, (list, tuple)):
                output_path = output_path[0]
            size = int(size)
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=max(1, size // 30),
                border=4,
            )
            qr.add_data(text)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color=fill_color, back_color=back_color)
            # Resize image to exact pixel size
            img = img.resize((size, size), Image.Resampling.NEAREST)
            img.save(output_path)
            
            return {"success": True, "size": os.path.getsize(output_path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_qr_base64(self, text, fill_color, back_color, size):
        try:
            size = int(size)
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=max(1, size // 30),
                border=4,
            )
            qr.add_data(text)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color=fill_color, back_color=back_color)
            img = img.resize((size, size), Image.Resampling.NEAREST)
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return {"success": True, "base64": f"data:image/png;base64,{img_str}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── APIs (Bypassing CORS completely!) ──
    def fetch_currency_rates(self):
        try:
            r = requests.get("https://api.exchangerate-api.com/v4/latest/EUR", timeout=8)
            r.raise_for_status()
            return {"success": True, "data": r.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def fetch_network_info(self, url):
        try:
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            return {"success": True, "data": r.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def fetch_ip_lookup(self, url):
        try:
            # We can also do plain IP Lookup bypasses
            r = requests.get(url, timeout=8)
            r.raise_for_status()
            # If the response is json, return it
            try:
                data = r.json()
            except:
                data = r.text
            return {"success": True, "data": data}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── PDF Studio APIs ──
    def pdf_merge(self, pdf_paths_json, output_path):
        try:
            from pypdf import PdfMerger
            import json
            pdf_paths = json.loads(pdf_paths_json)
            
            clean_paths = []
            for p in pdf_paths:
                if isinstance(p, list) or isinstance(p, tuple):
                    clean_paths.append(p[0])
                else:
                    clean_paths.append(p)
            
            if isinstance(output_path, list) or isinstance(output_path, tuple):
                output_path = output_path[0]
                
            merger = PdfMerger()
            for pdf in clean_paths:
                merger.append(pdf)
                
            merger.write(output_path)
            merger.close()
            return {"success": True}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def pdf_split(self, pdf_path, page_ranges, output_path):
        try:
            from pypdf import PdfReader, PdfWriter
            if isinstance(pdf_path, list) or isinstance(pdf_path, tuple):
                pdf_path = pdf_path[0]
            if isinstance(output_path, list) or isinstance(output_path, tuple):
                output_path = output_path[0]
                
            reader = PdfReader(pdf_path)
            writer = PdfWriter()
            
            total_pages = len(reader.pages)
            pages_to_keep = set()
            
            parts = page_ranges.replace(" ", "").split(",")
            for part in parts:
                if "-" in part:
                    start, end = part.split("-")
                    start_idx = max(0, int(start) - 1)
                    end_idx = min(total_pages, int(end))
                    for i in range(start_idx, end_idx):
                        pages_to_keep.add(i)
                else:
                    idx = int(part) - 1
                    if 0 <= idx < total_pages:
                        pages_to_keep.add(idx)
            
            for page_num in sorted(list(pages_to_keep)):
                writer.add_page(reader.pages[page_num])
                
            with open(output_path, "wb") as f:
                writer.write(f)
            return {"success": True}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def images_to_pdf(self, image_paths_json, output_path):
        try:
            import json
            image_paths = json.loads(image_paths_json)
            clean_paths = []
            for p in image_paths:
                if isinstance(p, list) or isinstance(p, tuple):
                    clean_paths.append(p[0])
                else:
                    clean_paths.append(p)
                    
            if isinstance(output_path, list) or isinstance(output_path, tuple):
                output_path = output_path[0]
                
            if not clean_paths:
                return {"success": False, "error": "Keine Bilder ausgewählt."}
                
            images = []
            first_img = None
            for idx, img_path in enumerate(clean_paths):
                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if idx == 0:
                    first_img = img
                else:
                    images.append(img)
                    
            if first_img:
                first_img.save(output_path, save_all=True, append_images=images)
                return {"success": True}
            return {"success": False, "error": "Fehler beim Laden der Bilder."}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def pdf_compress(self, pdf_path, output_path):
        try:
            from pypdf import PdfReader, PdfWriter
            if isinstance(pdf_path, list) or isinstance(pdf_path, tuple):
                pdf_path = pdf_path[0]
            if isinstance(output_path, list) or isinstance(output_path, tuple):
                output_path = output_path[0]
                
            reader = PdfReader(pdf_path)
            writer = PdfWriter()
            
            for page in reader.pages:
                page.compress_content_streams()
                writer.add_page(page)
                
            for page in writer.pages:
                for img in page.images:
                    try:
                        img.replace(img.image, quality=55)
                    except:
                        pass
                        
            with open(output_path, "wb") as f:
                writer.write(f)
            return {"success": True}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # ── File Compressor APIs ──
    def compress_image(self, image_path, output_path, scale_pct, quality, out_format):
        try:
            if isinstance(image_path, list) or isinstance(image_path, tuple):
                image_path = image_path[0]
            if isinstance(output_path, list) or isinstance(output_path, tuple):
                output_path = output_path[0]
                
            scale_pct = float(scale_pct) / 100.0
            quality = int(quality)
            
            img = Image.open(image_path)
            if out_format.upper() in ["JPG", "JPEG"] and img.mode in ("RGBA", "LA", "P"):
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[3] if img.mode == "RGBA" else None)
                img = bg
                
            if scale_pct != 1.0:
                w = int(img.width * scale_pct)
                h = int(img.height * scale_pct)
                img = img.resize((w, h), Image.Resampling.LANCZOS)
                
            pil_format = "JPEG" if out_format.upper() in ["JPG", "JPEG"] else out_format.upper()
            save_args = {}
            if pil_format in ["JPEG", "WEBP"]:
                save_args["quality"] = quality
                
            img.save(output_path, format=pil_format, **save_args)
            return {"success": True}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def compress_audio(self, audio_path, output_path, out_format, bitrate):
        try:
            if isinstance(audio_path, list) or isinstance(audio_path, tuple):
                audio_path = audio_path[0]
            if isinstance(output_path, list) or isinstance(output_path, tuple):
                output_path = output_path[0]
                
            cmd = [self._ffmpeg_path, '-y', '-i', audio_path, '-b:a', f"{bitrate}k", output_path]
            print("FFmpeg Audio Compress Command:", " ".join(cmd))
            
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.run(cmd, startupinfo=startupinfo, check=True)
            return {"success": True}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # ── Audio Recorder APIs ──
    def save_recorded_audio(self, base64_webm_data, out_format, output_path):
        try:
            import base64
            import tempfile
            
            if isinstance(output_path, list) or isinstance(output_path, tuple):
                output_path = output_path[0]
                
            if "," in base64_webm_data:
                base64_webm_data = base64_webm_data.split(",")[1]
            audio_bytes = base64.b64decode(base64_webm_data)
            
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
                
            try:
                cmd = [self._ffmpeg_path, '-y', '-i', temp_path]
                if out_format.lower() == 'mp3':
                    cmd.extend(['-c:a', 'libmp3lame', '-q:a', '2'])
                cmd.append(output_path)
                
                print("FFmpeg Audio Record Save Command:", " ".join(cmd))
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                subprocess.run(cmd, startupinfo=startupinfo, check=True)
                return {"success": True}
            finally:
                try: os.remove(temp_path)
                except: pass
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # ── Video Downloader APIs ──
    def analyze_video_url(self, url, browser="none"):
        try:
            import yt_dlp
            ydl_opts = {
                'extract_flat': True,
                'skip_download': True,
                'nocheckcertificate': True,
            }
            if browser and browser != "none":
                ydl_opts['cookiesfrombrowser'] = (browser,)
                
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unbekanntes Video')
                thumbnail = info.get('thumbnail', '')
                duration = info.get('duration', 0)
                
                if duration:
                    minutes = int(duration // 60)
                    seconds = int(duration % 60)
                    duration_str = f"{minutes}:{seconds:02d}"
                else:
                    duration_str = "Unbekannt"
                
                return {
                    "success": True,
                    "title": title,
                    "thumbnail": thumbnail,
                    "duration": duration_str
                }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def download_video_url(self, url, format_type, output_path, start_time="", end_time="", browser="none"):
        try:
            if isinstance(output_path, (list, tuple)):
                output_path = output_path[0]
            
            import yt_dlp
            
            trim_enabled = bool(start_time or end_time)
            if trim_enabled:
                base, ext = os.path.splitext(output_path)
                dl_path = base + "_full" + ext
            else:
                dl_path = output_path
            
            def progress_hook(d):
                if d['status'] == 'downloading':
                    total = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
                    downloaded = d.get('downloaded_bytes', 0)
                    if total > 0:
                        pct = int(downloaded / total * 100)
                        pct = min(99, pct) # Leave 100% for postprocessing completion
                        self._window.evaluate_js(f"if (typeof updateDownloadProgress === 'function') updateDownloadProgress({pct});")
                elif d['status'] == 'finished':
                    self._window.evaluate_js(f"if (typeof updateDownloadProgress === 'function') updateDownloadProgress(100);")
            
            ydl_opts = {
                'outtmpl': os.path.splitext(dl_path)[0] + '.%(ext)s',
                'progress_hooks': [progress_hook],
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'logtostderr': False,
                'quiet': True,
                'no_warnings': True,
            }
            if browser and browser != "none":
                ydl_opts['cookiesfrombrowser'] = (browser,)
            
            if self._ffmpeg_path and os.path.exists(self._ffmpeg_path):
                ydl_opts['ffmpeg_location'] = os.path.dirname(self._ffmpeg_path)
            
            format_type = format_type.lower()
            if format_type in ['mp3', 'm4a', 'wav']:
                ydl_opts['format'] = 'bestaudio/best'
                ydl_opts['postprocessors'] = [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': format_type,
                    'preferredquality': '192',
                }]
            else:
                ydl_opts['format'] = 'bestvideo+bestaudio/best'
                if format_type in ['mp4', 'mov', 'mkv', 'webm']:
                    ydl_opts['merge_output_format'] = format_type
                    ydl_opts['remux_video'] = format_type
                    if format_type in ['mp4', 'mov', 'mkv']:
                        ydl_opts['postprocessor_args'] = {
                            'ffmpeg': ['-c:a', 'aac']
                        }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                
            if trim_enabled:
                actual_dl_file = os.path.splitext(dl_path)[0] + "." + format_type
                if not os.path.exists(actual_dl_file):
                    for ext in ['.mp4', '.mkv', '.webm', '.mov', '.mp3', '.m4a', '.wav']:
                        test_path = os.path.splitext(dl_path)[0] + ext
                        if os.path.exists(test_path):
                            actual_dl_file = test_path
                            break
                if not os.path.exists(actual_dl_file):
                    raise FileNotFoundError(f"Downloaded file not found at {actual_dl_file}")
                
                cmd = [self._ffmpeg_path, '-y']
                if start_time:
                    cmd.extend(['-ss', start_time])
                if end_time:
                    cmd.extend(['-to', end_time])
                cmd.extend(['-i', actual_dl_file, '-c', 'copy', output_path])
                
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                subprocess.run(cmd, startupinfo=startupinfo, check=True)
                
                try:
                    os.remove(actual_dl_file)
                except:
                    pass
                
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def edit_video(self, input_path, output_path, start_time="", end_time="", format_type="mp4", resolution="original", mute=False, speed="1.0"):
        try:
            if isinstance(input_path, (list, tuple)): input_path = input_path[0]
            if isinstance(output_path, (list, tuple)): output_path = output_path[0]
            if isinstance(start_time, (list, tuple)): start_time = start_time[0]
            if isinstance(end_time, (list, tuple)): end_time = end_time[0]
            if isinstance(format_type, (list, tuple)): format_type = format_type[0]
            if isinstance(resolution, (list, tuple)): resolution = resolution[0]
            if isinstance(mute, (list, tuple)): mute = mute[0]
            if isinstance(speed, (list, tuple)): speed = speed[0]

            print(f"edit_video API: input={input_path}, output={output_path}, start={start_time}, end={end_time}, format={format_type}, resolution={resolution}, mute={mute}, speed={speed}")

            mute = bool(mute)
            speed_val = float(speed)
            
            # Check if input/output extensions are the same
            _, in_ext = os.path.splitext(input_path.lower())
            _, out_ext = os.path.splitext(output_path.lower())
            same_ext = (in_ext == out_ext)
            
            # Check if we can do stream copy (lightning fast)
            is_simple = (resolution == "original" and not mute and speed_val == 1.0 and same_ext)
            
            cmd = [self._ffmpeg_path, '-y']
            
            # Start/End times
            if start_time:
                cmd.extend(['-ss', start_time])
            if end_time:
                cmd.extend(['-to', end_time])
                
            cmd.extend(['-i', input_path])
            
            if is_simple:
                cmd.extend(['-c', 'copy', output_path])
            else:
                vf = []
                if speed_val != 1.0:
                    setpts = 1.0 / speed_val
                    vf.append(f"setpts={setpts}*PTS")
                if resolution:
                    resolution = resolution.strip()
                    if resolution != "original":
                        try:
                            w, h = resolution.split('x')
                            vf.append(f"scale={int(w)}:{int(h)}")
                        except:
                            pass
                
                if vf:
                    cmd.extend(['-vf', ",".join(vf)])
                
                if mute:
                    cmd.append('-an')
                else:
                    af = []
                    if speed_val != 1.0:
                        if 0.5 <= speed_val <= 2.0:
                            af.append(f"atempo={speed_val}")
                        else:
                            curr = speed_val
                            while curr > 2.0:
                                af.append("atempo=2.0")
                                curr /= 2.0
                            while curr < 0.5:
                                af.append("atempo=0.5")
                                curr /= 0.5
                            if curr != 1.0:
                                af.append(f"atempo={curr}")
                    if af:
                        cmd.extend(['-af', ",".join(af)])
                    
                    if out_ext == '.webm':
                        cmd.extend(['-c:a', 'libopus'])
                    else:
                        cmd.extend(['-c:a', 'aac'])
                
                if out_ext == '.webm':
                    cmd.extend(['-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0', '-preset', 'ultrafast', output_path])
                else:
                    cmd.extend(['-c:v', 'libx264', '-preset', 'fast', output_path])
            
            print("FFmpeg Command:", " ".join(cmd))
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            subprocess.run(cmd, startupinfo=startupinfo, check=True)
            
            # Fetch output metadata to confirm the resolution changes
            out_w, out_h = 0, 0
            size_bytes = 0
            if os.path.exists(output_path):
                try:
                    cap = cv2.VideoCapture(output_path)
                    if cap.isOpened():
                        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    size_bytes = os.path.getsize(output_path)
                except Exception as ex:
                    print("Failed to read output metadata:", ex)
            
            return {
                "success": True,
                "width": out_w,
                "height": out_h,
                "size_bytes": size_bytes
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def get_weather(self, lat, lon):
        try:
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            r = requests.get(weather_url, timeout=5)
            r.raise_for_status()
            wdata = r.json()
            
            curr = wdata.get("current_weather", {})
            temp = curr.get("temperature")
            code = curr.get("weathercode", 0)
            is_day = curr.get("is_day", 1)
            windspeed = curr.get("windspeed")
            winddirection = curr.get("winddirection")
            
            condition_info = WMO_CODES.get(code, {"en": "Unknown", "de": "Unbekannt", "emoji": "🌡️"})
            
            return {
                "success": True,
                "temp": temp,
                "emoji": condition_info["emoji"],
                "desc_en": condition_info["en"],
                "desc_de": condition_info["de"],
                "is_day": is_day,
                "windspeed": windspeed,
                "winddirection": winddirection
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_local_weather(self):
        try:
            geo_data = None
            geo_errors = []
            
            # Try freeipapi.com
            try:
                r = requests.get("https://freeipapi.com/api/json", timeout=4)
                if r.status_code == 200:
                    data = r.json()
                    geo_data = {
                        "lat": data.get("latitude"),
                        "lon": data.get("longitude"),
                        "city": data.get("cityName"),
                        "country_code": data.get("countryCode")
                    }
            except Exception as e:
                geo_errors.append(f"freeipapi: {e}")
                
            if not geo_data:
                # Try ipapi.co
                try:
                    r = requests.get("https://ipapi.co/json/", timeout=4)
                    if r.status_code == 200:
                        data = r.json()
                        geo_data = {
                            "lat": data.get("latitude"),
                            "lon": data.get("longitude"),
                            "city": data.get("city"),
                            "country_code": data.get("country_code")
                        }
                except Exception as e:
                    geo_errors.append(f"ipapi: {e}")
                    
            if not geo_data:
                # Try ipwho.is
                try:
                    r = requests.get("https://ipwho.is/", timeout=4)
                    if r.status_code == 200:
                        data = r.json()
                        if data.get("success"):
                            geo_data = {
                                "lat": data.get("latitude"),
                                "lon": data.get("longitude"),
                                "city": data.get("city"),
                                "country_code": data.get("country_code")
                            }
                except Exception as e:
                    geo_errors.append(f"ipwho: {e}")
                    
            if not geo_data or geo_data.get("lat") is None or geo_data.get("lon") is None:
                return {
                    "success": False, 
                    "error": f"Location detection failed. Details: {', '.join(geo_errors)}"
                }
                
            # Fetch weather
            wres = self.get_weather(geo_data["lat"], geo_data["lon"])
            if not wres["success"]:
                return wres
                
            # Merge geodata
            wres["city"] = geo_data["city"] or "Unknown"
            wres["country_code"] = geo_data["country_code"] or ""
            wres["lat"] = geo_data["lat"]
            wres["lon"] = geo_data["lon"]
            return wres
            
        except Exception as e:
            return {"success": False, "error": str(e)}

if __name__ == '__main__':
    # Cleanup any old temp files in the gui/tools folder
    try:
        tools_dir = get_resource_path('gui/tools')
        if os.path.exists(tools_dir):
            for f in os.listdir(tools_dir):
                if (f.startswith('temp_video_') or f.startswith('temp_preview_')) and f.endswith('.mp4'):
                    try: os.remove(os.path.join(tools_dir, f))
                    except: pass
    except Exception as e:
        print("Cleanup failed:", e)

    # Initialize API
    api = ToolForgeAPI()
    
    # Load index file
    html_file = get_resource_path('gui/ToolForge.html')
    
    # Check if we are in packaged environment and local folder exists
    if not os.path.exists(html_file):
        print(f"Error: GUI entry point not found at {html_file}")
        sys.exit(1)

    # Create window
    window = webview.create_window(
        title='ToolForge Desktop',
        url=html_file,
        js_api=api,
        width=1220,
        height=850,
        min_size=(900, 600),
        resizable=True,
        text_select=True,
        background_color='#080810'
    )
    
    # Inject window instance to API after window is loaded to prevent race conditions
    def on_loaded():
        api._set_window(window)
    window.events.loaded += on_loaded
    
    # Save settings on close handler
    def on_closing():
        print("ToolForge window is closing. Ensuring settings are saved...")
        
    window.events.closing += on_closing
    
    # Start app
    webview.start(debug=False, private_mode=True)


