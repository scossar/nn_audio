#include "m_pd.h"
#include <math.h>
#include <stdlib.h>

/*
* notes:
* - echo_state_network.md
* - pd_audio_rate_neural_network.md
* - upsample_and_downsample_concepts.md
* - downsample_click_artifacts.md
* - downsample_audio_glitch.md
* - downsampling_solved.md (somewhat solved)
*/

// #define XTRASAMPS 4

t_class *down_class = NULL;

typedef struct _down {
  t_object x_obj;

  t_sample *x_downsampled_buffer;
  int x_downsampled_buffer_size; // a power of 2; e.g. 2^14
  int x_write_phase;
  int x_last_written;
  int x_downsample_factor; // a power of 2; e.g. 2^1

  double x_read_phase;
  double x_read_phase_inc;

  // if a filter was added:
  t_sample x_state;
  t_float x_sr;

  t_inlet *x_signal_inlet;
  t_outlet *x_downsampled_outlet;
  t_float x_f; // dummy var for CLASS_MAINSIGNALIN
} t_down;

static void initialize_buffers(t_down *x) {
  x->x_downsampled_buffer = getbytes(sizeof(t_float)
                                     * (x->x_downsampled_buffer_size));
  if (!x->x_downsampled_buffer) {
    pd_error(x, "down~: failed to allocate memory to downsample buffer");
    return;
  }
}

static void down_free(t_down *x) {
  if (x->x_signal_inlet) {
    inlet_free(x->x_signal_inlet);
  }
  if (x->x_downsampled_outlet) {
    outlet_free(x->x_downsampled_outlet);
  }

  if (x->x_downsampled_buffer) {
    freebytes(x->x_downsampled_buffer, sizeof(t_float) *
              (x->x_downsampled_buffer_size));
    x->x_downsampled_buffer = NULL;
  }
}

static t_int *down_perform(t_int *w) {
  t_down *x = (t_down *)(w[1]);
  t_sample *in1 = (t_sample *)(w[2]);
  t_sample *out1 = (t_sample *)(w[3]);
  int n = (int)(w[4]);

  int downsample_buffer_samps = x->x_downsampled_buffer_size; // a power of 2
  int db_mask = x->x_downsampled_buffer_size - 1;
  int df = x->x_downsample_factor; // a power of 2
  int df_mask = df - 1;

  int write_phase = x->x_write_phase;
  double read_phase = x->x_read_phase;

  t_sample *buffer = x->x_downsampled_buffer;

  while (n--) {
    t_sample f = *in1++;
    t_sample output = (t_sample)0.0;

    if ((n & df_mask) == 0) {
      buffer[write_phase] = f;
      write_phase = (write_phase + 1) & db_mask;
    }

    int iread_phase = (int)read_phase;
    double frac = read_phase - (double)iread_phase;
    t_sample d1 = buffer[iread_phase];
    t_sample d2 = buffer[(iread_phase + 1) & db_mask];
    output = d1 + frac * (d2 - d1);

    *out1++ = output;

    read_phase = (read_phase + x->x_read_phase_inc);
    while (read_phase >= downsample_buffer_samps) read_phase -= downsample_buffer_samps;
  }
  x->x_read_phase = read_phase;
  x->x_write_phase = write_phase;

  return (w+5);
}

static void down_dsp(t_down *x, t_signal **sp) {
  dsp_add(down_perform, 4, x, sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_length);
}

static void *down_new(void) {
  t_down *x = (t_down *)pd_new(down_class);

  x->x_downsampled_buffer = NULL;
  // x->x_downsampled_buffer_size = 8192; // 2^13
  x->x_downsampled_buffer_size = 16384; // 2^14
  // x->x_downsampled_buffer_size = 262144; // 2^18
  x->x_downsample_factor = 4; // power of 2
  x->x_read_phase = (double)0.0;
  x->x_write_phase = 0;
  x->x_last_written = 0;
  x->x_read_phase_inc = (double)((double)1.0/(double)x->x_downsample_factor);
  // for reading back at downsampled rate. note there are glitches with this
  // approach
  // x->x_read_phase_inc = (double)((double)1.0/((double)x->x_downsample_factor * (double)x->x_downsample_factor));

  initialize_buffers(x);

  x->x_signal_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  x->x_downsampled_outlet = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

void down_tilde_setup(void) {
  down_class = class_new(gensym("down~"),
                         (t_newmethod)down_new,
                         (t_method)down_free,
                         sizeof(t_down),
                         CLASS_DEFAULT,
                         0);
  class_addmethod(down_class, (t_method)down_dsp,
                  gensym("dsp"), A_CANT, 0);
  CLASS_MAINSIGNALIN(down_class, t_down, x_f);
}


