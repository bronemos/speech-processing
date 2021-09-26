import subprocess
import sys

def update_pkg(package):
    subprocess.call([sys.executable, "-m", "pip", "install", '--upgrade', package])

#Run update
if __name__ == '__main__':
    update_pkg('scipy')
    update_pkg('numpy')
    update_pkg('matplotlib')
    print('Packages Updated for Exercise 1')