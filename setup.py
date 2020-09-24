from setuptools import setup

setup(name='pricer',
      version='0.0.1',
      description='Option Pricing, Risk Management and Visualisation tools',
      author='...',
      author_email='...',
      packages=['visualizer', 'models'],
      install_requires=['matplotlib',
			'numpy',
                        'plotly',
			'scipy'])

