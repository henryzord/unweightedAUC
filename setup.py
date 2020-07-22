import numpy
from distutils.core import setup, Extension

metrics_modulename = Extension(
    'PBIL.metrics',  # module name, as it will appear when importing insing python (e.g. import <module name>)
    sources=['src/metrics.cpp'],
    include_dirs=[numpy.get_include()],
    language='c++',
    extra_compile_args=['-std=c++11']
)


setup(
    name="PBIL",  # library name, as it will appear when using conda list and pip list
    version="0.1",
    package_dir={'PBIL': '.'},
    packages=["PBIL"],
    description="PBILC is a library that aims to find  good voting ensemble configurations.",
    ext_modules=[metrics_modulename],
    install_requires=['numpy', 'pandas'], # 'joblib>=0.13.2', 'PyHamcrest>=1.9.0'],
    author="Henry E. L. Cagnini",
    author_email="henry.cagnini@acad.pucrs.br",
    keywords="AUTO-ML, Machine Learning, Estimation of Distribution Algorithms",
    license="BSD 3-Clause License", 
    url="https://github.com/henryzord/eacomp",
    # package_data={'PBILC': ['grammar/*']},
    # include_package_data=True,
)





