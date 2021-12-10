if [ "$1" = "" ]
then
    echo -e "\n${YELLOW}Warning: Not using GPU ${NC}\n"
else
    echo -e "\n${GREEN}Using CUDA device $1 ${NC}\n"
fi

export CUDA_VISIBLE_DEVICES=$1
export PYTHONPATH=$PWD:$PYTHONPATH
