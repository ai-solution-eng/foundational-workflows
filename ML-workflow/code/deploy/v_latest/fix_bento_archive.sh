export BENTOML_MODEL_TAG=$1
export DESTINATION_PATH=$(pwd)/bento_inspect_folder
export BENTO_DEST_NAME=medmnist_cnn_demo-bento-v-latest.bento
export BENTO_FILE_PATH="tmp_bento.bento"

mkdir $DESTINATION_PATH
cd $DESTINATION_PATH

bentoml export $BENTOML_MODEL_TAG ${DESTINATION_PATH}/${BENTO_FILE_PATH}
tar -xvf tmp_bento.bento -C .
rm tmp_bento.bento 

cp ../install.sh ./env/python/
cp ../requirements.txt ./env/python/

rm ./env/python/requirements.in

cd ..

tar -cvf ${BENTO_DEST_NAME} -C bento_inspect_folder .

echo "Bento file exported correctly with name ${BENTO_DEST_NAME}"
cd ../../
