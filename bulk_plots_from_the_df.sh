for line in $(cat $1)
do 
echo "$line"
python plots_from_the_df.py /storage/Nicholas/Data/pixel_classification/"$line"_pixels.csv $line
done 