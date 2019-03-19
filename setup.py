from setuptools import setup

setup(name='shusaku',
      version='0.1',
      description='Go IA',
      author='Samuel Batissou, Mathieu Pont',
      license='GPL',
      packages=['shusaku'],
      install_requires=[
          'libshusaku',
          'tensorflow', 'numpy'
      ],
      zip_safe=False
  )
