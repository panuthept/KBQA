from setuptools import setup, find_packages


# with open("requirements.txt") as f:
#     dependencies = [line for line in f]

setup(
    name='kbqa',
    version='1.0',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    license='Apache-2.0 License',
    author='Panuthep Tasawong',
    author_email='panuthep.t_s20@vistec.ac.th',
    description='Question Answering over Knowledge Base (KBQA)',
    python_requires='>=3.11',
    # install_requires=dependencies
)