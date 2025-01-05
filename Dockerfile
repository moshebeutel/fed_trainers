FROM python:3.8-slim

WORKDIR /workspace

COPY ./biolab_utilities /workspace
COPY ./gpytorch /workspace
COPY ./pyeeg /workspace
COPY ./pypolyagamma /workspace


COPY ./emg_utils.py /workspace
COPY ./gep_utils.py /workspace
COPY ./hyper_sweeper.sh /workspace
COPY ./model.py /workspace
COPY ./utils.py /workspace

COPY ./sweeper_cifar10_dp_no_gp.py /workspace
COPY ./sweeper_cifar10_gep_public_no_gp.py /workspace
COPY ./sweeper_cifar10_gep_no_gp.py /workspace

COPY ./trainer_cifar10_dp_no_gp.py /workspace
COPY ./trainer_cifar10_gep_public_no_gp.py /workspace
COPY ./trainer_cifar10_gep_no_gp.py /workspace

COPY ./trainer_putEMG_dp_no_gp.py /workspace
COPY ./trainer_putEMG_gep_no_gp.py /workspace
COPY ./trainer_putEMG_gep_public_no_gp.py /workspace

COPY ./trainer_sgd_dp_no_gp.py /workspace
COPY ./trainer_gep_public_no_gp.py /workspace
COPY ./trainer_gep_private_no_gp.py /workspace


COPY ./requirements.txt /workspace

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "trainer_putEMG_gep_no_gp.py"]

