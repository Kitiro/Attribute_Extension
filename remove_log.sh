data='AWA AWA2 SUN APY CUB'
for set in $data
do
    rm -rf output/$set/*/*.png
    rm -rf output/$set/*/*.log
done
