from setuptools import setup, find_packages

setup(
    name='rice_price_prediction',
    version='0.1.0',
    author='Lillian Phyo',
    author_email='khinpyaephyosan@gmail.com',
    description='A project for predicting rice prices using LSTM models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lillainphyo/rice_price_prediction',
    packages=find_packages(include=['utils', 'utils.*']),
    install_requires=[
        'flask',
        'mlflow',
        'tensorflow',
        'numpy',
        'pandas',
        'matplotlib',
        'psutil',
        'pytest',
    ],
    entry_points={
        'console_scripts': [
            'train-model=utils.main:main',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)