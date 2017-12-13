for file in ./*.xml; do
    name=${file:2:${#file}-6}
    xml_extension=".xml"
    mid_extension=".mid"
    echo $name$xml_extension
    echo $name$mid_extension
    /Applications/MuseScore\ 2.app/Contents/MacOS/mscore $name$xml_extension -o $name$mid_extension
done
