from setuptools import setup

setup(name='shusaku',
      version='0.1',
      description='e world',
      author='Samuel Batissou',
      license='GPL',
      packages=['shusaku'],
      install_requires=[
          'rusty_goban',
      ],
      zip_safe=False
  )
