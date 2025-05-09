#include "m_pd.h"
#include <math.h>

/*
* notes:
* - echo_state_network.md
* - pd_audio_rate_neural_network.md
* - upsample_and_downsample_concepts.md
* - downsample_click_artifacts.md
*/

t_class *updown_class = NULL;

// #define XTRA_SAMPS 1

typedef struct _updown {
  t_object x_obj;
  t_float *x_full_buffer;
  t_float *x_downsampled_buffer;
  int x_full_buffer_size;
  int x_downsampled_buffer_size;
  int x_downsample_factor;

  int x_input_single_phase;
  int x_input_downsample_phase;
  int x_downsample_single_phase;

  double x_upsample_phase_inc;
  double x_downsample_upsample_phase;

  t_float x_filter_state;
  t_float x_filter_cutoff;
  t_float x_alpha;

  t_float x_sr;
  t_float x_f; // dummy variable for CLASS_MAINSIGNALIN

  t_inlet *x_input_inlet;
  t_outlet *x_output_outlet;
} t_updown;

static void initilize_buffers(t_updown *x) {
  x->x_full_buffer = getbytes(sizeof(t_float) * (x->x_full_buffer_size));
  if (!x->x_full_buffer) {
    pd_error(x, "updown~: unable to assign memory to full buffer");
    return;
  }

  x->x_downsampled_buffer = getbytes(sizeof(t_float) * x->x_downsampled_buffer_size);
  if (!x->x_downsampled_buffer) {
    pd_error(x, "updown~: unable to assign memory to downsample buffer");
    return;
  }
}

static t_int *updown_perform(t_int *w) {
  t_updown *x = (t_updown *)(w[1]);
  t_sample *in1 = (t_sample *)(w[2]);
  t_sample *out1 = (t_sample *)(w[3]);
  int n = (int)(w[4]);

  t_float *full_buff = x->x_full_buffer; // size 2^16
  t_float *downsample_buff = x->x_downsampled_buffer; // size 2^14

  int downsample_factor = x->x_downsample_factor; // 8
  int input_buffer_mask = x->x_full_buffer_size - 1;
  int downsample_buffer_mask = x->x_downsampled_buffer_size - 1;

  int input_single_phase = x->x_input_single_phase;
  int input_downsample_phase = x->x_input_downsample_phase;
  int downsample_single_phase = x->x_downsample_single_phase;
  double downsample_upsample_phase = x->x_downsample_upsample_phase;
  downsample_upsample_phase -= n;
  if (downsample_upsample_phase < 0) downsample_upsample_phase += x->x_downsampled_buffer_size;
  double upsample_inc = x->x_upsample_phase_inc; // 1.0 / downsample_factor


  while (n--) {
    t_sample buff_in = *in1++;
    t_sample filtered_in = x->x_filter_state + x->x_alpha * buff_in - x->x_filter_state;
    full_buff[input_single_phase] = filtered_in;
    x->x_filter_state = filtered_in;


    downsample_buff[downsample_single_phase] = full_buff[input_downsample_phase];
    int iupsample_phase = downsample_upsample_phase;
    t_sample frac = downsample_upsample_phase - (t_sample)iupsample_phase;
    t_sample us1 = downsample_buff[iupsample_phase];
    t_sample us2 = downsample_buff[(iupsample_phase + 1) & downsample_buffer_mask];

    *out1++ = us1 + frac * (us2 - us1);
    // *out1++ = downsample_buff[downsample_single_phase];

    input_single_phase = (input_single_phase + 1) & input_buffer_mask;
    // i'm guessing the issue is here: 
    input_downsample_phase = (input_downsample_phase + downsample_factor) & input_buffer_mask;
    downsample_single_phase = (downsample_single_phase + 1) & downsample_buffer_mask;
    downsample_upsample_phase = downsample_upsample_phase + upsample_inc;
    downsample_upsample_phase = (double)(((int)downsample_upsample_phase) & downsample_buffer_mask) +
      (downsample_upsample_phase - (int)downsample_upsample_phase);
  }

  x->x_input_single_phase = input_single_phase;
  x->x_input_downsample_phase = input_downsample_phase;
  x->x_downsample_single_phase = downsample_single_phase;
  x->x_downsample_upsample_phase = downsample_upsample_phase;

  return (w+5);
}

static void updown_dsp(t_updown *x, t_signal **sp) {
  x->x_sr = sp[0]->s_sr;
  dsp_add(updown_perform, 4, x, sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_length);
}


static void updown_free(t_updown *x) {
  if (x->x_full_buffer) {
    freebytes(x->x_full_buffer, sizeof(t_float) * (x->x_full_buffer_size));
    x->x_full_buffer = NULL;
  }
  if (x->x_downsampled_buffer) {
    freebytes(x->x_downsampled_buffer, sizeof(t_float) * (x->x_downsampled_buffer_size));
    x->x_downsampled_buffer = NULL;
  }

  if (x->x_input_inlet) {
    inlet_free(x->x_input_inlet);
  }

  if (x->x_output_outlet) {
    outlet_free(x->x_output_outlet);
  }
}

static void *updown_new(void) {
  t_updown *x = (t_updown *)pd_new(updown_class);

  x->x_full_buffer = NULL;
  x->x_downsampled_buffer = NULL;

  x->x_full_buffer_size = 65536; // 2^16
  // x->x_downsampled_buffer_size = 8192; // 2^13
  x->x_downsampled_buffer_size = 32768; // 2^15
  // x->x_downsample_factor = 8; // (2^16) / (2^13)
  x->x_downsample_factor = 2; // (2^16) / (2^15)
  x->x_sr = (t_float)0.0;

  x->x_input_single_phase = 0;
  x->x_input_downsample_phase = 0;
  x->x_downsample_single_phase = 0;
  x->x_downsample_upsample_phase = 0.0;
  x->x_upsample_phase_inc = (double)(1.0 / (double)x->x_downsample_factor);

  x->x_filter_state = (t_float)0.0;
  x->x_filter_cutoff = (t_float)0.5 / (t_float)x->x_downsample_factor;
  x->x_alpha = (t_float)(1.0 - exp(-2.0 * M_PI * x->x_filter_cutoff));

  initilize_buffers(x);

  x->x_input_inlet = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
  x->x_output_outlet = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

void updown_tilde_setup(void) {
  updown_class = class_new(gensym("updown~"),
                           (t_newmethod)updown_new,
                           (t_method)updown_free,
                           sizeof(t_updown),
                           CLASS_DEFAULT,
                           0);

  class_addmethod(updown_class, (t_method)updown_dsp,
                  gensym("dsp"), A_CANT, 0);
  CLASS_MAINSIGNALIN(updown_class, t_updown, x_f);
}
