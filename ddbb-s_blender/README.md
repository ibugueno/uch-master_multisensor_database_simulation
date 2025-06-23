# Setting Up a Conda Environment from `requirements.txt`

This guide provides instructions on how to create a Conda environment named `blender` with Python 3.10.13, using a `requirements.txt` file to install dependencies.

---

## Steps to Create the Environment

1. **Create the Conda environment**  
   Open a terminal and run the following command to create a Conda environment named `blender` with Python 3.10.13:

   conda create --name blender python=3.10.13

2. **Activate the environment**  
   After the environment is created, activate it:

   conda activate blender

3. **Install dependencies using `requirements.txt`**  
   Ensure the `requirements.txt` file is in your current working directory, then run:

   pip install -r requirements.txt

   This will install all the packages listed in the `requirements.txt` file into the `blender` environment.

---

## Execution

1. **Move to scripts folder**

2. Execute

python 1_main_blender_render_back.py --sensor asus --scene all --luminosities 1000 --background --save_labels
python 1_main_blender_render_back.py --sensor davis346 --scene all --luminosities 1000 --background --save_labels
python 1_main_blender_render_back.py --sensor evk4 --scene all --luminosities 1000 --background --save_labels

---

## Additional Notes

- **Verify the Python version**  
  After activating the environment, confirm the Python version to ensure it matches `3.10.13`:

  python --version

- **Check installed packages**  
  Once the installation is complete, you can verify the installed packages:

  pip list

- **Deactivate the environment**  
  To exit the `blender` environment, simply run:

  conda deactivate

---

## Troubleshooting

- **Missing dependencies**: If some dependencies fail to install, ensure that `requirements.txt` contains valid package names and versions.
- **Mixing Conda and Pip**: Be aware that some packages may conflict if they are installed via both Conda and Pip. Always prefer Conda for packages available in Conda repositories.

## Blender version

Blender 4.0.2