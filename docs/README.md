# To release a new version

First, update version to `0.x.y` in `setup.py` and push to master.
Then, install necessary packages:

```bash
pip install --upgrade setuptools wheel twine
```

Make a tag:

```bash
git tag v0.x.y
git push origin v0.x.y
```

Make a wheel:

```bash
python3 setup.py sdist bdist_wheel
```

Upload to pypi:

```bash
twine upload dist/*0.x.y*
```

Note that `autograd` doesn't install correctly on test.pypi.org,
so `viabel` will not install correctly from test.pypi.org, even when it will work
on pypi.
