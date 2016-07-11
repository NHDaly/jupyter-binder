
# Echo the IP address + port of the docker instance to connect to:
echo "----- Jupyter Notebook hosted at: http://`docker-machine ip default`:8888 -----"
echo "----- FOR GUI, don't forget to RUN THIS COMMAND in Xterm: -----"
echo "socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\""

DOCKER_VM_DISPLAY_IP=`ifconfig | pcregrep -M "vboxnet0.*\n.*\n.*" | pcregrep -o "[1-9]+\.[1-9]+\.[1-9]+\.[1-9]+" | head -1`


# Start the "mybinder" docker instance, and connect to it.
#  - Mount the repo inside the container.
#  - Run interactively and expose port 8888.
#  - Also forward the DISPLAY to the X11 server, listening from xterm!
#  - Start the notebook and accept connections from outside the container (0.0.0.0).
docker run \
  -v `pwd`:/home/main/binder \
  -it -p 8888:8888 \
  -e DISPLAY="$DOCKER_VM_DISPLAY_IP:0" \
  mybinder \
  ./start-notebook.sh "--ip=0.0.0.0"

