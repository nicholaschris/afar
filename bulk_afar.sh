foldername=$(date +%Y%m%d)
mkdir -p  /storage/Nicholas/Data/"$foldername"
for line in $(cat $1)
do 
echo "$line"
python afar.py /storage/Nicholas/Data/ $line /storage/Nicholas/Data/"$foldername"/
done 
