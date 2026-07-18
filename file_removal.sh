cd /cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/data_files/power_spectra/kappa_galaxy/L1000N1800/HYDRO_LOW_SIGMA8/Blue/lightcone0/

for filename in *.txt; do
    [ -f "$filename" ] || continue

    # Skip the files we want to keep
    [[ "$filename" == *_non_rotated.txt ]] && continue

    rm "$filename"
done

cd /cosma8/data/dp004/dc-conl1/FLAMINGO/patchy_screening/