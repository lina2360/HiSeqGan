This folder contains:

- a map trained with the Matlab SOMToolbox (http://www.cis.hut.fi/somtoolbox/), with the following files:
-- iris.sompak.data: input data in SOMPAK format (input vector and class information, see e.g. http://www.cis.hut.fi/projects/somtoolbox/package/docs2/som_read_data.html)
-- iris.sompak.unit: trained map in SOMPAK format (only unit information)

The map can be trained by "makeMatlabSOM.m"


- the same map converted to SOMLib Format, in the files iris.somlib.unit, iris.somlib.wgt, and iris.somlib.dwm. 
The map can be re-converted by running convertSOMPAK2SOMLib.sh

- the same map converted to ESOM Format (http://databionic-esom.sourceforge.net/user.html#File_formats), in the files iris.esom.bm, iris.esom.cls, iris.esom.lrn, iris.esom.names, iris.esom.wts
The map can be re-converted by running convertSOMPAK2ESOM.sh

- screenshots of several visualisations taking from the Matlab SOMToolbox in combination with the IFS SOMVis suite, as well as the ESOM
 