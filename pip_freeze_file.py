import subprocess

def generate_requirements_file():
    # Execute pip freeze command to get installed packages
    try:
        output = subprocess.check_output(['pip', 'freeze']).decode('utf-8')
        # Write the output to requirements.txt
        with open('requirements.txt', 'w') as f:
            f.write(output)
        print("requirements.txt file has been generated successfully.")
    except subprocess.CalledProcessError as e:
        print("Error:", e.output)

if __name__ == "__main__":
    generate_requirements_file()