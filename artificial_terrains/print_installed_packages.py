import sys
import subprocess


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

    # Print installed packages
    print("Installed Packages:")
    installed_packages = get_installed_packages()
    print(installed_packages)


if __name__ == "__main__":
    main()
