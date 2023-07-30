#include "datasets.h"
#include "util/rng.h"

#include <math.h> // sinf, cosf

f32 offsets[100] = {
 -1.0856306f,   0.99734545f,  0.2829785f,  -1.50629471f, -0.57860025f,  1.65143654f,
 -2.42667924f, -0.42891263f,  1.26593626f, -0.8667404f,  -0.67888615f, -0.09470897f,
  1.49138963f, -0.638902f,   -0.44398196f, -0.43435128f,  2.20593008f,  2.18678609f,
  1.0040539f,   0.3861864f,   0.73736858f,  1.49073203f, -0.93583387f,  1.17582904f,
 -1.25388067f, -0.6377515f,   0.9071052f,  -1.4286807f,  -0.14006872f, -0.8617549f,
 -0.25561937f, -2.79858911f, -1.7715331f,  -0.69987723f,  0.92746243f, -0.17363568f,
  0.00284592f,  0.68822271f, -0.87953634f,  0.28362732f, -0.80536652f, -1.72766949f,
 -0.39089979f,  0.57380586f,  0.33858905f, -0.01183049f,  2.39236527f,  0.41291216f,
  0.97873601f,  2.23814334f, -1.29408532f, -1.03878821f,  1.74371223f, -0.79806274f,
  0.02968323f,  1.06931597f,  0.89070639f,  1.75488618f,  1.49564414f,  1.06939267f,
 -0.77270871f,  0.79486267f,  0.31427199f, -1.32626546f,  1.41729905f,  0.80723653f,
  0.04549008f, -0.23309206f, -1.19830114f,  0.19952407f,  0.46843912f, -0.83115498f,
  1.16220405f, -1.09720305f, -2.12310035f,  1.03972709f, -0.40336604f, -0.12602959f,
 -0.83751672f, -1.60596276f,  1.25523737f, -0.68886898f,  1.66095249f,  0.80730819f,
 -0.31475815f, -1.0859024f,  -0.73246199f, -1.21252313f,  2.08711336f,  0.16444123f,
  1.15020554f, -1.26735205f,  0.18103513f,  1.17786194f, -0.33501076f,  1.03111446f,
 -1.08456791f, -1.36347154f,  0.37940061f, -0.37917643f
};

f32* linspace(f32 x1, f32 x2, usize n)
{
    f32* result = calloc(n, sizeof(f32));
    f32 dx = (x2 - x1) / (n - 1.0f);

    for (usize i = 0; i < n; ++i)
        result[i] = x1 + ((f32)i * dx);
    
    return result;
}

// NOTE(lucas): Spiral dataset adapted from https://github.com/Sentdex/nnfs/blob/master/nnfs/datasets/spiral.py
void create_spiral_data(usize samples, u8 classes, mat_f32* out_data, mat_u8* out_labels)
{
    *out_data = mat_f32_zeros(samples*classes, 2);
    *out_labels = mat_u8_zeros(samples*classes, 1);

    for (u8 class = 0; class < classes; ++class)
    {
        // TODO(lucas): linspace would be useful
        f32* r = linspace(0.0f, 1.0f, samples);
        f32* t = linspace((f32)class*4.0f, ((f32)class + 1)*4.0f, samples);

        for (usize i = 0; i < samples; ++i)
        {
            t[i] += offsets[i] * 0.2f;
            
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
