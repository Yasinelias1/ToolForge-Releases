import os
import sys
import subprocess
import shutil

def main():
    print("=== Start ToolForge Desktop Build Process ===")
    
    # Paths
    project_dir = os.path.abspath(os.path.dirname(__file__))
    venv_dir = os.path.join(project_dir, "venv")
    pyinstaller_path = os.path.join(venv_dir, "Scripts", "pyinstaller.exe")
    
    if not os.path.exists(pyinstaller_path):
        print(f"Error: PyInstaller not found at {pyinstaller_path}")
        sys.exit(1)

    # Copy face cascade xml from venv/Lib/site-packages/cv2/data/ to project root
    cv2_data_dir = os.path.join(venv_dir, "Lib", "site-packages", "cv2", "data")
    alt2_cascade = os.path.join(cv2_data_dir, "haarcascade_frontalface_alt2.xml")
    dest_alt2 = os.path.join(project_dir, "haarcascade_frontalface_alt2.xml")
    if os.path.exists(alt2_cascade):
        print(f"Copying {alt2_cascade} to project root...")
        shutil.copy2(alt2_cascade, dest_alt2)
    else:
        print(f"Error: Face cascade xml not found at {alt2_cascade}")
        sys.exit(1)

    # Locate static-ffmpeg binaries in virtual environment
    ffmpeg_bin_dir = os.path.join(venv_dir, "Lib", "site-packages", "static_ffmpeg", "bin")
    if not os.path.exists(ffmpeg_bin_dir):
        print(f"Error: ffmpeg binaries not found at {ffmpeg_bin_dir}. Please run main.py once first.")
        sys.exit(1)

    # Locate insightface data objects (e.g. meanshape_68.pkl)
    insightface_data_dir = os.path.join(venv_dir, "Lib", "site-packages", "insightface", "data")
    insightface_objects_dir = os.path.join(insightface_data_dir, "objects")

    # Build command
    cmd = [
        pyinstaller_path,
        "--clean",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name=ToolForge",
        f"--icon={os.path.join(project_dir, 'logo.ico')}",
        # Add the entire gui folder to the packaged EXE
        f"--add-data={os.path.join(project_dir, 'gui')};gui",
        # Add the face detection XML file
        f"--add-data={os.path.join(project_dir, 'haarcascade_frontalface_alt2.xml')};.",
        # Add the ffmpeg binaries to the expected site-packages path inside sys._MEIPASS
        f"--add-data={ffmpeg_bin_dir};static_ffmpeg/bin",
        # Add the insightface data package
        f"--add-data={insightface_data_dir};insightface/data",
        # Add the objects folder to the root of sys._MEIPASS for frozen environments
        f"--add-data={insightface_objects_dir};objects",
        os.path.join(project_dir, "main.py")
    ]
    
    print("Running PyInstaller Command:")
    print(" ".join(cmd))
    
    # Execute build
    try:
        subprocess.run(cmd, check=True)
        print("\n=== Build Completed Successfully! ===")
        print(f"Your packaged folder is located at: {os.path.join(project_dir, 'dist', 'ToolForge')}")
    except subprocess.CalledProcessError as e:
        print(f"\nError: PyInstaller exited with error code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
