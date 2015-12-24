for line in $(cat $1)
do 
echo "$line"
python create_list_of_pixels.py $line
done 
