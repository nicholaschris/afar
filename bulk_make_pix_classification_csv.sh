for line in $(cat $1)
do 
echo "$line"
python make_pix_classification_csv.py /storage/Nicholas/Data/ $line
done 