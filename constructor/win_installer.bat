@ECHO OFF
echo ""
echo "######################################################"
echo "##### The main installation of PanSeg starts now #####"
echo "######################################################"
echo ""
CALL "%PREFIX%\Scripts\activate.bat"
cd "%PREFIX%"
tar xf build.gz
CALL conda install -y -c "%PREFIX%\conda_bld" -c conda-forge panseg
