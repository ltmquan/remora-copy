mainzisegmap_295816_dim1, 1, 1)
void mainzisegmap_295816(__global int *global_failure, int64_t index_primexp_295803, int64_t ctx_param_ext_444476, __global unsigned char *mem_444474, __global unsigned char *mem_param_444480, __global unsigned char *mem_444482)
{
    #define segmap_tblock_sizze_295809 (mainzisegmap_295816zisegmap_tblock_sizze_295809)
    if (*global_failure >= 0)
        return;
    
    int32_t local_tid_464478;
    int32_t tblock_sizze_464481;
    int32_t wave_sizze_464480;
    int32_t block_id_464479;
    int32_t global_tid_464477;
    int64_t phys_tid_295816;
    int64_t global_tid_464482;
    int64_t slice_464483;
    int64_t slice_464484;
    int64_t slice_464485;
    int64_t slice_464486;
    int64_t gtid_295812;
    int64_t remnant_464487;
    int64_t gtid_295813;
    int64_t remnant_464488;
    int64_t gtid_295814;
    int64_t remnant_464489;
    int64_t gtid_295815;
    int64_t remnant_464490;
    
    local_tid_464478 = get_local_id(0);
    tblock_sizze_464481 = get_local_size(0);
    wave_sizze_464480 = LOCKSTEP_WIDTH;
    block_id_464479 = get_tblock_id(0);
    global_tid_464477 = block_id_464479 * tblock_sizze_464481 + local_tid_464478;
    phys_tid_295816 = sext_i32_i64(global_tid_464477);
    global_tid_464482 = sext_i32_i64(block_id_464479) * segmap_tblock_sizze_295809 + sext_i32_i64(local_tid_464478);
    slice_464483 = (int64_t) 19;
    slice_464484 = (int64_t) 19 * slice_464483;
    slice_464485 = (int64_t) 5 * slice_464484;
    slice_464486 = (int64_t) 512 * slice_464485;
    gtid_295812 = squot64(global_tid_464482, slice_464485);
    remnant_464487 = global_tid_464482 - gtid_295812 * slice_464485;
    gtid_295813 = squot64(remnant_464487, slice_464484);
    remnant_464488 = remnant_464487 - gtid_295813 * slice_464484;
    gtid_295814 = squot64(remnant_464488, slice_464483);
    remnant_464489 = remnant_464488 - gtid_295814 * slice_464483;
    gtid_295815 = remnant_464489;
    remnant_464490 = remnant_464489 - gtid_295815;
    if (((slt64(gtid_295812, (int64_t) 512) && slt64(gtid_295813, (int64_t) 5)) && slt64(gtid_295814, (int64_t) 19)) && slt64(gtid_295815, (int64_t) 19)) {
        float x_295817;
        float x_295818;
        float binop_295819;
        
        x_295817 = ((__global float *) mem_param_444480)[gtid_295812 * ctx_param_ext_444476 + gtid_295813 * (int64_t) 361 + gtid_295814 * (int64_t) 19 + gtid_295815];
        x_295818 = ((__global float *) mem_444474)[gtid_295812 * (int64_t) 9025 + index_primexp_295803 * (int64_t) 1805 + gtid_295813 * (int64_t) 361 + gtid_295814 * (int64_t) 19 + gtid_295815];
        binop_295819 = fmax32(x_295817, x_295818);
        ((__global float *) mem_444482)[gtid_295812 * (int64_t) 1805 + gtid_295813 * (int64_t) 361 + gtid_295814 * (int64_t) 19 + gtid_295815] = binop_295819;
    }
    
  error_0:
    return;
    #undef segmap_tblock_sizze_295809
}
