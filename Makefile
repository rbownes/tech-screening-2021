setup:
    @curl -O 'http://files.grouplens.org/datasets/movielens/ml-25m.zip'
    @unzip ml-25m.zip
    @python3 -m venv env
    @source env/bin/activate
    @pip3 install requirements.txt