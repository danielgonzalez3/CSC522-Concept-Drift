FROM python:3.11

# Make APT non-interactive
RUN echo 'APT::Get::Assume-Yes "true";' > /etc/apt/apt.conf.d/99semaphore
RUN echo 'DPkg::Options "--force-confnew";' >> /etc/apt/apt.conf.d/99semaphore
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

RUN apt update && apt upgrade
RUN echo 'Acquire::Check-Valid-Until no;' >> /etc/apt/apt.conf

RUN ln -sf /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN locale-gen C.UTF-8 || true

RUN pip install --upgrade pip setuptools
RUN mkdir -p /var/lib/src
WORKDIR /var/lib/src

COPY . /var/lib/src
RUN pip install -r requirements.txt