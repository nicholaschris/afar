for line in $(cat $1)
do 
echo "$line"
python parse_mtl.py /storage/Nicholas/Data/ $line 
done 
