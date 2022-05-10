import nox
from nox.sessions import Session


@nox.session(reuse_venv=True, python="3.9")
def black(session: Session) -> None:
    # session.install('black')
    if session.posargs:
        pathes = session.posargs
    else:
        pathes = ["src", "tests", "noxfile.py"]
    session.run("black", *pathes, external=True)


@nox.session(reuse_venv=True, python="3.9")
def mypy(session: Session) -> None:
    # session.install('mypy')
    if session.posargs:
        pathes = session.posargs
    else:
        pathes = ["src", "tests", "noxfile.py"]
    session.run("mypy", *pathes, external=True)


@nox.session(reuse_venv=False, python="3.9")
def tests(session: Session) -> None:
    # session.install('pytest')
    session.run("pytest", "tests", external=True)


@nox.session(reuse_venv=True, python="3.9")
def flake8(session: Session) -> None:
    # session.install('flake8')
    if session.posargs:
        pathes = session.posargs
    else:
        pathes = ["src", "tests", "noxfile.py"]
    session.run("flake8", "--append-config", "flake8.ini", *pathes, external=True)
