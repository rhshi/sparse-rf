import setuptools

setuptools.setup(
    name="sparse_rf",
    install_requires=[
      'numpy',
      'spgl1',
      'matplotlib',
      'scikit-learn',
      'jupyter',
      'namedlist'
      # 'jax',
      # 'jaxlib==0.1.65+cuda110' 
    ],
    zip_safe=False,
)