@ECHO OFF
echo ""
echo "######################################################"
echo "##### The main installation of PanSeg starts now #####"
echo "######################################################"
echo ""
CALL "%PREFIX%\Scripts\activate.bat"
cd "%PREFIX%"
tar xf build.gz
CALL conda install -y "%PREFIX%\conda_bld::panseg" -c conda-forge
