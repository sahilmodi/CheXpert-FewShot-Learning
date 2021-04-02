data_path=$1/CheXpert-v1.0-small
zip_path=$data_path.zip
# curl http://download.cs.stanford.edu/deep/CheXpert-v1.0-small.zip --output $zip_path
unzip -q $zip_path -d $1
rm $zip_path
echo Saved to $data_path
python utils/split_data.py --path $data_path