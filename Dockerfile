FROM python:3.8-slim

WORKDIR /workspace

COPY ./biolab_utilities /workspace
COPY ./gpytorch /workspace
COPY ./pyeeg /workspace
COPY ./pypolyagamma /workspace


COPY fed_trainers/datasets/emg_utils.py /workspace
COPY fed_trainers/trainers/gep/gep_utils.py /workspace
COPY ./hyper_sweeper.sh /workspace
COPY fed_trainers/trainers/model.py /workspace
COPY fed_trainers/trainers/utils.py /workspace

COPY fed_trainers/sweepers/dp_sgd/sweeper_cifar10_dp_no_gp.py /workspace
COPY fed_trainers/sweepers/gep/sweeper_cifar10_gep_public_no_gp.py /workspace
COPY fed_trainers/sweepers/gep/sweeper_cifar10_gep_no_gp.py /workspace

COPY fed_trainers/trainers/dp_sgd/trainer_cifar10_dp_no_gp.py /workspace
COPY fed_trainers/trainers/gep/trainer_cifar10_gep_public_no_gp.py /workspace
COPY fed_trainers/trainers/gep/trainer_cifar10_gep_no_gp.py /workspace

COPY fed_trainers/trainers/dp_sgd/trainer_putEMG_dp_no_gp.py /workspace
COPY fed_trainers/trainers/gep/trainer_putEMG_gep_no_gp.py /workspace
COPY fed_trainers/trainers/gep/trainer_putEMG_gep_public_no_gp.py /workspace

COPY fed_trainers/trainers/dp_sgd/trainer_sgd_dp_no_gp.py /workspace
COPY fed_trainers/trainers/gep/trainer_gep_public_no_gp.py /workspace
COPY fed_trainers/trainers/gep/trainer_gep_private_no_gp.py /workspace


COPY ./requirements.txt /workspace

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "trainer_putEMG_gep_no_gp.py"]

