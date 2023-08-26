#include "datasets.h"
#include "math/cnnfs_math.h"
#include "util/rng.h"

#include <math.h> // sinf, cosf

// NOTE(lucas): Spiral dataset adapted from https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py
void create_spiral_data(usize samples, u8 classes, mat_f32* out_data, mat_u8* out_labels)
{
    *out_data = mat_f32_zeros(samples*classes, 2);
    *out_labels = mat_u8_zeros(samples*classes, 1);

    for (u8 class = 0; class < classes; ++class)
    {
        f32* r = linspace(0.0f, 1.0f, samples);
        f32* t = linspace((f32)class*4.0f, ((f32)class + 1)*4.0f, samples);

        for (usize i = 0; i < samples; ++i)
        {
            t[i] += rand_f32_gauss(0.0f, 1.0f) * 0.2f;
            
            // For each class, offset starting row by the number of samples, then add the index
            usize row = samples*class + i;

            // TODO(lucas): set_row, set_col operations for matrices
            f32 sample_x = r[i] * sinf(t[i]*2.5f);
            f32 sample_y = r[i] * cosf(t[i]*2.5f);

            mat_f32_set_val(out_data, row, 0, sample_x);
            mat_f32_set_val(out_data, row, 1, sample_y);
            mat_u8_set_val(out_labels, row, 0, class);
        }
    }
}
