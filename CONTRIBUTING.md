# CONTRIBUTING

## Virtual Env setup

```shell
$ python3 -m venv venv
$ source activate venv/bin/activate
(venv)$ pip install -U pip
(venv)$ pip install opencv-python==4.1.1.26
(venv)$ pip install -r requirements.txt
(venv)$ pip install -r requirements-dev.txt
(venv)$ pip install -e module
```

## Docker maintenance

```shell
# Build:
(venv) $ make -f docker.mk build
...
# Test the shell:
(venv) $ make -f docker.mk start
...
```

## Code maintenance

```shell
# pylint the code:
(venv) $ make -f tasks.mk pylint
...
# pep8 the code:
(venv) $ make -f tasks.mk pep8
...
```

