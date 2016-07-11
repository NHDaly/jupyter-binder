
# Echo the IP address + port of the docker instance to connect to:
echo "----- Jupyter Notebook hosted at: http://`docker-machine ip default`:8888 -----"


# Start the "mybinder" docker instance, and connect to it.
#  - Mount the repo inside the container.
#  - Run interactively and expose port 8888.
#  - expose the x11vnc port as well.
#  - Start the notebook and accept connections from outside the container (0.0.0.0).
docker run \
  -v `pwd`:/home/main/binder \
  -it -p 8888:8888 \
  -p 5900 \
  mybinder \
  x11vnc -forever -create

