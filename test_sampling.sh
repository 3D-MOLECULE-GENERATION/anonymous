#for id in range(100):
#    !python scripts/sample_diffusion.py configs/sampling.yml --data_id {id} 
# 7,14,16,22,36,51,63,77,80,81
for id in {0..97}
#for id in {7,14,16,22,36,51,63,77,80,81}
do 
    python scripts/sample_diffusion.py configs/sampling_Frag.yml --data_id $id
done     