mkdir assets
cd assets
python ../script/_download.py

# background_texture
unzip background_texture.zip

# embodiments
unzip embodiments.zip

# objects
unzip objects.zip

cd ..
echo "making vidar related file"
cp -r assets/embodiments/aloha-agilex assets/embodiments/aloha-vidar
cp vidar_assets/*.yml  assets/embodiments/aloha-vidar
cp vidar_assets/arx5_description_isaac.urdf assets/embodiments/aloha-vidar/urdf


echo "Configuring Path ..."
python ./script/update_embodiment_config_path.py