FROM co1lin/ubuntu-basic:latest

RUN set -ex; source ~/miniforge3/bin/activate; mamba create -y -n cweval python=3.10; rm -rf /home/ubuntu/.cache

COPY requirements.txt .

# Install python deps
RUN set -ex; source ~/miniforge3/bin/activate; conda activate cweval; \
    pip install -r requirements.txt; \
    rm -rf /home/ubuntu/.cache

COPY run_tests.py .
COPY run_tests.sh .
COPY recheck-linux-x64 .
COPY pytest.ini .

# Command to run when container starts
CMD ["sleep", "infinity"]