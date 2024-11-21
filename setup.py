from setuptools import setup
from setuptools import find_packages
#import pip

# initial source code from: https://github.com/xtianmcd/GCNeuro


#%cd /content/gdrive/MyDrive/Colab/Workshop/torchquantum
#!pip3 install --editable .
# if quantum models

#git clone https://github.com/google/jax
#cd jax
#pip install jaxlib


setup(name='SR_XAI',
      version='0.1',
      description='Deep Learning 3D XAI and SR framework analysis in Python',
      url='',
      author='Michail Mamalakis',
      author_email='mm2703@cam.ac.uk',
      license='GPL-3.0+',
      packages=['SR_XAI'],
      install_requires=[
         
          'wandb',         
	  'torch>=1.10',
	  'nibabel',
          'unfoldNd',
          'torch_geometric',
          'scikit-image',
          'qiskit',
          'torchvision',
          'torchdata',
          'matplotlib==3.3.4',
	  'quantus',
	  'scikit-learn',
	  'scipy',
          'captum',
          'monai',
          'pandas',
          'seaborn',
          'torchmetrics',
          'einops',
	],
      zip_safe=False
)


