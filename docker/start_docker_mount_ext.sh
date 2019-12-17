# instructions for running this the first time:
# pull the docker image from docker hub: run ` docker pull febert/recplan:latest `

docker run  -v $RECPLAN_DATA_DIR/:/workspace/recplan_data \
                   -v $RECPLAN_EXP_DIR/:/workspace/experiments \
		   -v $ROBOSUITE_DIR/:/mount/robosuite \
                   -v /raid/:/raid \
                   -v /:/parent \
                   --name=${ROBO_DOCKER_NAME} \
-e RECPLAN_DATA_DIR=/workspace/recplan_data \
-e RECPLAN_EXP_DIR=/workspace/experiments \
-t -d \
--shm-size 8G \
febert/recplan:latest \
/bin/bash
docker exec ${ROBO_DOCKER_NAME} /bin/bash -c "cd /mount/robosuite; python setup.py develop"

# orybkin/recplan:latest  was used before
