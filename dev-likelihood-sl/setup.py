from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["full_likelihood.pyx"],
        compiler_directives={'language_level' : "3", 'boundscheck' : False, 'wraparound' : False, 'nonecheck' : False, 'cdivision' : True},
        # compiler_directives={'language_level' : "3"},
        annotate=True)

    # ext_modules = cythonize("global_defs.pyx", #"event_rate.pyx",
        # compiler_directives={'language_level' : "3"}, annotate=True)

    # ext_modules = cythonize("cosmology.pyx", #"event_rate.pyx",
        # compiler_directives={'language_level' : "3"}, annotate=True)

    # include_path = [numpy.get_include()]
    # ext_modules = cythonize("src/*.pyx", include_path=include_path annotate=True)
)
