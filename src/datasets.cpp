#include "datasets.h"
#include "math/cnnfs_math.h"
#include "util/rng.h"

// NOTE(lucas): Spiral dataset adapted from https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py
void create_spiral_data(size samples, u32 classes, mat<f32>* out_data, vec<u32>* out_labels, Device device)
{
    *out_data = mat_zeros<f32>(samples*classes, 2, device);
    *out_labels = vec_zeros<u32>(samples*classes, device);

    for (u32 i = 0; i < classes; ++i)
    {
        size offset = samples*i;

        // Increasing radius over the domain results in a spiral shape rather than a circle
        vec<f32> r = linspace(0.0f, 1.0f, samples, device);

        vec<f32> rand_vals = vec_rand_gauss_standard(samples);
        vec_to(&rand_vals, device);
        vec_scale(rand_vals, 0.2f);

        // Each class has a different domain of angles, which rotates the spiral of each class.
        vec<f32> t_range = linspace((f32)i*4.0f, (f32)(i+1)*4.0f, samples, device);
        // Adding random values to the angles adds noise to the output so the spiral shapes are imperfect.
        t_range += rand_vals;
        vec_scale(t_range, 2.5f);

        vec<f32> sample_x = sin_vec(t_range);
        vec<f32> sample_y = cos_vec(t_range);
        vec_had(sample_x, r);
        vec_had(sample_y, r);

        vec<u32> labels = vec_full<u32>(samples, i, device);
        
        mat_set_col_range(*out_data, sample_x, 0, offset);
        mat_set_col_range(*out_data, sample_y, 1, offset);
        vec_set_range(*out_labels, labels, offset);
    }
}
