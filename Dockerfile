FROM python:3.6
MAINTAINER Micah Melling, micahmelling@gmail.com
RUN groupadd docker
RUN useradd --create-home appuser
RUN usermod -aG docker appuser
RUN newgrp docker
RUN mkdir /app
WORKDIR /app
COPY . /app
RUN chown -R appuser:docker /app
ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN pip install --no-cache-dir -r requirements.txt
USER appuser
EXPOSE 8000
RUN chmod a+rx run.sh
CMD ["./run.sh"]