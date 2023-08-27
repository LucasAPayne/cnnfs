#include "datasets.h"
#include "math/cnnfs_math.h"
#include "util/rng.h"

// NOTE(lucas): Spiral dataset adapted from https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py
void create_spiral_data(usize samples, u8 classes, mat_f32* out_data, mat_u8* out_labels)
{
    *out_data = mat_f32_zeros(samples*classes, 2);
    *out_labels = mat_u8_zeros(samples*classes, 1);

    for (u8 class = 0; class < classes; ++class)
    {
        vec_f32 r = linspace(0.0f, 1.0f, samples);

        vec_f32 rand_vals = vec_f32_scale(randn_f32_gauss(0.0f, 1.0f, samples), 0.2f);
        vec_f32 t_range = linspace((f32)class*4.0f, ((f32)class + 1)*4.0f, samples);
        vec_f32 t = vec_f32_add(t_range, rand_vals);

        for (usize i = 0; i < samples; ++i)
        {
            // For each class, offset starting row by the number of samples, then add the index
            usize row = samples*class + i;

            // TODO(lucas): set_row, set_col operations for matrices
            f32 sample_x = r.data[i] * sin_f32(t.data[i]*2.5f);
            f32 sample_y = r.data[i] * cos_f32(t.data[i]*2.5f);

            mat_f32_set_val(out_data, row, 0, sample_x);
            mat_f32_set_val(out_data, row, 1, sample_y);
            mat_u8_set_val(out_labels, row, 0, class);
        }
    }
}
