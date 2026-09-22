echo ""
echo "######################################################"
echo "##### The main installation of PanSeg starts now #####"
echo "######################################################"
echo ""
. "${PREFIX}/etc/profile.d/conda.sh" && conda activate "${PREFIX}"
cd "${PREFIX}"
tar xf build.gz
conda install -y "${PREFIX}/conda_bld::panseg" -c conda-forge
