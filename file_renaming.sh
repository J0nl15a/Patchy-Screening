cd /cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/galaxy_galaxy/L1000N3600/HYDRO_FIDUCIAL/Blue/lightcone0/;

for filename in *.txt; do 
    [ -f "$filename" ] || continue
    mv "$filename" "${filename//.txt/}_non_rotated.txt"
done

cd /cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/kappa_galaxy/L1000N3600/HYDRO_FIDUCIAL/Blue/lightcone0/;

for filename in *.txt; do 
    [ -f "$filename" ] || continue
    mv "$filename" "${filename//.txt/}_non_rotated.txt"
done

cd /cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/;