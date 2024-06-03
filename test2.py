import sys
import subprocess
import os


def get_installed_packages():
    try:
        # Using subprocess to get the list of installed packages
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'],
                                capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error retrieving installed packages: {e}"


def main():
    # Print Python version
    python_version = sys.version
    print("Python Version:")
    print(python_version)
    print()

    # Print path to the Python executable
    python_executable = sys.executable
    print("Path to Python Executable:")
    print(python_executable)
    print()

    # Print sys.path
    print("sys.path:")
    for path in sys.path:
        print(path)
    print()

    # Print installed packages
    print("Installed Packages:")
    installed_packages = get_installed_packages()
    print(installed_packages)

    # Ensure site-packages is in sys.path
    site_packages_path = os.path.join(os.path.dirname(sys.executable), 'lib', 'site-packages')
    if site_packages_path not in sys.path:
        print(f"Adding {site_packages_path} to sys.path")
        sys.path.append(site_packages_path)

    # Attempt to import scipy
    try:
        import scipy
        print("scipy imported successfully")
    except ModuleNotFoundError as e:
        print(f"Error importing scipy: {e}")


if __name__ == "__main__":
    main()
