../../../somtoolbox.sh MapFileFormatConverter --inputFormat SOMPak iris.wgt.sompak --outputFormat SOMLib iris
../../../somtoolbox.sh DataMapper -w iris.wgt.gz -v ../../datasets/iris.vec
mv iris.remapped.unit.gz iris.unit.gz
mv iris.remapped.dwm.gz iris.dwm.gz

