import nox

@nox.session(reuse_venv=True, python='3.9')
def black(session):
    #session.install('black')
    if session.posargs:
        folders = session.posargs
    else:
        folders = ['src', 'tests']
    session.run('black',
         *folders,
         external=True)

@nox.session(reuse_venv=True, python='3.9')
def mypy(session):
    #session.install('mypy')
    if session.posargs:
        folders = session.posargs
    else:
        folders = ['src', 'tests']
    session.run('mypy',
         *folders,
         external=True)

@nox.session(reuse_venv=False, python='3.9')
def tests(session):
    #session.install('pytest')
    session.run('pytest',
         'tests',
         external=True)

@nox.session(reuse_venv=True, python='3.9')
def flake8(session):
    #session.install('flake8')
    if session.posargs:
        folders = session.posargs
    else:
        folders = ['src', 'tests']
    session.run('flake8',
            '--append-config',
            'flake8.ini',
            *folders,
            external=True)