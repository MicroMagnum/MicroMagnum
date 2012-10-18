
for file in `find . -type f -name "*.eps"`; do
	png=${file%.eps}.png
	pdf=${file%.eps}.pdf
	echo Converting $file to $png
	convert -density 200 $file $png
done

