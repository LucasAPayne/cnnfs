#include "datasets.h"
#include "math/cnnfs_math.h"
#include "util/rng.h"

// NOTE(lucas): Spiral dataset adapted from https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py
void create_spiral_data(usize samples, u8 classes, mat_f32* out_data, vec_u8* out_labels)
{
    *out_data = mat_f32_zeros(samples*classes, 2);
    *out_labels = vec_u8_zeros(samples*classes);

    for (u8 class = 0; class < classes; ++class)
    {
        usize offset = samples*class;

        vec_f32 r = linspace(0.0f, 1.0f, samples);

        vec_f32 rand_vals = vec_f32_scale(rand_vec_f32_gauss(0.0f, 1.0f, samples), 0.2f);
        vec_f32 t_range = linspace((f32)class*4.0f, ((f32)class + 1)*4.0f, samples);
        vec_f32 t = vec_f32_scale(vec_f32_add(t_range, rand_vals), 2.5f);

        vec_f32 sample_x = vec_f32_had(sin_vec_f32(t), r);
        vec_f32 sample_y = vec_f32_had(cos_vec_f32(t), r);

        vec_u8 class_vec = vec_u8_full(samples, class);
        
        mat_f32_set_col_range(out_data, sample_x, 0, offset);
        mat_f32_set_col_range(out_data, sample_y, 1, offset);
        vec_u8_set_range(out_labels, class_vec, offset);
    }
}
