from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "garch_est",
        ["cpp/garch-estimate.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++"
    ),
]

setup(
    name="garch_est",
    version="0.0.1",
    ext_modules=ext_modules,
)