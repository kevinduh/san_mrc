SCRIPTPATH=$( cd $(dirname $0) ; pwd -P )
IMAGE=allenlao/pytorch-allennlp-rtd # docker image
#IMAGE=allenlao/pytorch-allennlp-rt # docker image
#IMAGE=allenlao/pytorch-allennlp-v2 # docker image
#IMAGE=allenlao/pytorch-allennlp # docker image
#IMAGE=allenlao/pytorchv4 # docker image


echo $SCRIPTPATH
# export CUDA_SO=$(\ls /usr/lib/x86_64-linux-gnu/libcuda.* | xargs -I{} echo '-v {}:{}')
# export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')

echo $SCRIPTPATH
echo $CUDA_SO
echo $DEVICES

# start docker
nvidia-docker run \
--rm -it \
--net host \
--volume $SCRIPTPATH:/san_mrc \
--volume /home/xiaodl/my/data/:/cs \
--interactive --tty $IMAGE /bin/bash
