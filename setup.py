import setuptools

setuptools.setup(
    name="sparse_rf",
    install_requires=[
      'numpy',
      'spgl1',
      'matplotlib'    
    ],
    zip_safe=False,
)