mainzisegmap_304454_dim1, 1, 1)
void mainzisegmap_304454(__global int *global_failure, __global unsigned char *mem_450668, __global unsigned char *mem_450693)
{
    #define segmap_tblock_sizze_304447 (mainzisegmap_304454zisegmap_tblock_sizze_304447)
    if (*global_failure >= 0)
        return;
    
    int32_t local_tid_472253;
    int32_t tblock_sizze_472256;
    int32_t wave_sizze_472255;
    int32_t block_id_472254;
    int32_t global_tid_472252;
    int64_t phys_tid_304454;
    int64_t global_tid_472257;
    int64_t slice_472258;
    int64_t slice_472259;
    int64_t slice_472260;
    int64_t slice_472261;
    int64_t gtid_304450;
    int64_t remnant_472262;
    int64_t gtid_304451;
    int64_t remnant_472263;
    int64_t gtid_304452;
    int64_t remnant_472264;
    int64_t gtid_304453;
    int64_t remnant_472265;
    
    local_tid_472253 = get_local_id(0);
    tblock_sizze_472256 = get_local_size(0);
    wave_sizze_472255 = LOCKSTEP_WIDTH;
    block_id_472254 = get_tblock_id(0);
    global_tid_472252 = block_id_472254 * tblock_sizze_472256 + local_tid_472253;
    phys_tid_304454 = sext_i32_i64(global_tid_472252);
    global_tid_472257 = sext_i32_i64(block_id_472254) * segmap_tblock_sizze_304447 + sext_i32_i64(local_tid_472253);
    slice_472258 = (int64_t) 19;
    slice_472259 = (int64_t) 19 * slice_472258;
    slice_472260 = (int64_t) 5 * slice_472259;
    slice_472261 = (int64_t) 512 * slice_472260;
    gtid_304450 = squot64(global_tid_472257, slice_472260);
    remnant_472262 = global_tid_472257 - gtid_304450 * slice_472260;
    gtid_304451 = squot64(remnant_472262, slice_472259);
    remnant_472263 = remnant_472262 - gtid_304451 * slice_472259;
    gtid_304452 = squot64(remnant_472263, slice_472258);
    remnant_472264 = remnant_472263 - gtid_304452 * slice_472258;
    gtid_304453 = remnant_472264;
    remnant_472265 = remnant_472264 - gtid_304453;
    if (((slt64(gtid_304450, (int64_t) 512) && slt64(gtid_304451, (int64_t) 5)) && slt64(gtid_304452, (int64_t) 19)) && slt64(gtid_304453, (int64_t) 19)) {
        float map_304457;
        float redout_418357 = -1.0e30F;
        
        for (int64_t i_418358 = 0; i_418358 < (int64_t) 5; i_418358++) {
            float x_304458;
            float binop_304461;
            float redout_tmp_472266;
            
            x_304458 = ((__global float *) mem_450668)[gtid_304450 * (int64_t) 9025 + i_418358 * (int64_t) 1805 + gtid_304451 * (int64_t) 361 + gtid_304452 * (int64_t) 19 + gtid_304453];
            binop_304461 = fmax32(x_304458, redout_418357);
            redout_tmp_472266 = binop_304461;
            redout_418357 = redout_tmp_472266;
        }
        map_304457 = redout_418357;
        ((__global float *) mem_450693)[gtid_304450 * (int64_t) 1805 + gtid_304451 * (int64_t) 361 + gtid_304452 * (int64_t) 19 + gtid_304453] = map_304457;
    }
    
  error_0:
    return;
    #undef segmap_tblock_sizze_304447
}
