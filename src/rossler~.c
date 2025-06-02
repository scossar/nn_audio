#include "m_pd.h"
#include <math.h>

/**
 * notes:
 * - rossler_attractor.md
 **/

typedef struct _rossler {
  t_object x_obj;

  t_float x_conv; // 1 / sample_rate

  // state variables
  t_float x_x, x_y, x_z;
  t_float x_x_init, x_y_init, x_z_init;

  // params
  t_float x_a, x_b, x_c;

  int x_previous_phase_sector;
  t_float x_z_pulse_threshold;
  int x_phase_pulse_sectors;
  t_sample x_predicted_z_amp;

  t_float x_2_recip;
  t_float x_6_recip;

  t_float x_f;

  t_outlet *x_out_x, *x_out_y, *x_out_z, *x_out_xy_pulse, *x_out_z_pulse;
  // t_outlet *x_phase_sector_bang, *x_new_cycle_bang;
} t_rossler;

static t_class *rossler_class = NULL;

static void *rossler_new(t_symbol *s, int argc, t_atom *argv) {
  t_rossler *x = (t_rossler *)pd_new(rossler_class);

  t_float x_x, x_y, x_z;
  if (argc < 3) {
    x_x = (t_float)1.0;
    x_y = (t_float)2.0;
    x_z = (t_float)3.0;
  } else {
    x_x = atom_getfloat(argv++);
    x_y = atom_getfloat(argv++);
    x_z = atom_getfloat(argv++);
  }

  x->x_x = x_x;
  x->x_y = x_y;
  x->x_z = x_z;

  x->x_x_init = x_x;
  x->x_y_init = x_y;
  x->x_z_init = x_z;

  x->x_a = (t_float)0.1;
  x->x_b = (t_float)0.1;
  x->x_c = (t_float)14.0;

  x->x_previous_phase_sector = (int)0;
  x->x_z_pulse_threshold = (t_float)0.05;
  x->x_phase_pulse_sectors = (int)1;
  // note: not implmented for now:
  x->x_predicted_z_amp = (t_sample)1.0;

  x->x_conv = (t_float)0.0;
  x->x_f = (t_float)1.0;

  x->x_2_recip = (t_float)0.5;
  x->x_6_recip = (t_float)((t_float)1.0 / (t_float)6.0);

  x->x_out_z_pulse = outlet_new(&x->x_obj, &s_signal);
  x->x_out_xy_pulse = outlet_new(&x->x_obj, &s_signal);
  x->x_out_z = outlet_new(&x->x_obj, &s_signal);
  x->x_out_y = outlet_new(&x->x_obj, &s_signal);
  x->x_out_x = outlet_new(&x->x_obj, &s_signal);
  // x->x_new_cycle_bang = outlet_new(&x->x_obj, &s_bang);
  // x->x_phase_sector_bang = outlet_new(&x->x_obj, &s_bang);

  return (void *)x;
}

static void rossler_free(t_rossler *x) {
  if (x->x_out_x) outlet_free(x->x_out_x);
  if (x->x_out_y) outlet_free(x->x_out_y);
  if (x->x_out_z) outlet_free(x->x_out_z);
  if (x->x_out_xy_pulse) outlet_free(x->x_out_xy_pulse);
  if (x->x_out_z_pulse) outlet_free(x->x_out_z_pulse);
  // if (x->x_phase_sector_bang) outlet_free(x->x_phase_sector_bang);
  // if (x->x_new_cycle_bang) outlet_free(x->x_new_cycle_bang);
}

static void rossler_derivatives(t_rossler *x, t_float x_x, t_float x_y, t_float x_z,
                                t_float *dx, t_float *dy, t_float *dz) {
  t_float a = x->x_a;
  t_float b = x->x_b;
  t_float c = x->x_c;

  *dx = -x_y - x_z;
  *dy = x_x + a * x_y;
  *dz = b + x_z * (x_x - c);
}

static int get_phase_sector(t_float x, t_float y) {
  int sector = 0;
  if (y >= 0) {
    if (x >= 0) {
      sector = (y > x) ? 1 : 0;
    } else {
      sector = (-x > y) ? 3 : 2;
    }
  } else {
    if (x < 0) {
      sector = (-y > -x) ? 5 : 4;
    } else {
      sector = (x > -y) ? 7 : 6;
    }
  }
  return sector;
}

static t_int *rossler_perform(t_int *w) {
  t_rossler *x = (t_rossler *)(w[1]);
  t_sample *in_f = (t_sample *)(w[2]);
  t_sample *out_x = (t_sample *)(w[3]);
  t_sample *out_y = (t_sample *)(w[4]);
  t_sample *out_z = (t_sample *)(w[5]);
  t_sample *out_xy_pulse = (t_sample *)(w[6]);
  t_sample *out_z_pulse = (t_sample *)(w[7]);
  int n = (int)(w[8]);

  t_float conv = x->x_conv;
  t_float recip2 = x->x_2_recip; // 1/2
  t_float recip6 = x->x_6_recip; // 1/6

  t_float x_x = x->x_x;
  t_float x_y = x->x_y;
  t_float x_z = x->x_z;

  int previous_phase = x->x_previous_phase_sector;
  int new_phase = 0;
  // int phase_bang = 0;
  t_float z_pulse_threshold = x->x_z_pulse_threshold;
  int pulse_sectors = x->x_phase_pulse_sectors;

  // see notes:
  // t_float radius = sqrtf(x_x * x_x + x_y * x_y);
  t_float predicted_z_amp = x->x_predicted_z_amp;

  while (n--) {
    t_sample f = *in_f++;

    t_float dt = f * conv;
    // just a guess at this point. maybe it can go faster
    if (dt > (t_float)0.01) dt = (t_float)0.01;
    if (dt <= 0) dt = (t_float)1e-9;

    t_float k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
    t_float x_temp, y_temp, z_temp;
    t_float dt_half = dt * recip2;

    rossler_derivatives(x, x_x, x_y, x_z, &k1x, &k1y, &k1z);
    x_temp = x_x + k1x * dt_half;
    y_temp = x_y + k1y * dt_half;
    z_temp = x_z + k1z * dt_half;

    rossler_derivatives(x, x_temp, y_temp, z_temp, &k2x, &k2y, &k2z);
    x_temp = x_x + k2x * dt_half;
    y_temp = x_y + k2y * dt_half;
    z_temp = x_z + k2z * dt_half;

    rossler_derivatives(x, x_temp, y_temp, z_temp, &k3x, &k3y, &k3z);
    x_temp = x_x + k3x * dt;
    y_temp = x_y + k3y * dt;
    z_temp = x_z + k3z * dt;

    rossler_derivatives(x, x_temp, y_temp, z_temp, &k4x, &k4y, &k4z);

    x_x += dt * (k1x + (t_float)2.0*k2x + (t_float)2.0*k3x + k4x) * recip6;
    x_y += dt * (k1y + (t_float)2.0*k2y + (t_float)2.0*k3y + k4y) * recip6;
    x_z += dt * (k1z + (t_float)2.0*k2z + (t_float)2.0*k3z + k4z) * recip6;

    t_sample xy_pulse = (t_sample)0.0;
    t_sample z_pulse = (t_sample)0.0;
    new_phase = get_phase_sector(x_x, x_y);
    if (new_phase < pulse_sectors) xy_pulse = (t_sample)1.0;

    // trying to predict the amplitude here, but just hardcoded for now (see
    // notes):
    z_pulse = (x_z > z_pulse_threshold) ? predicted_z_amp : (t_sample)0.0;

    *out_x++ = x_x;
    *out_y++ = x_y;
    *out_z++ = x_z;
    *out_xy_pulse++ = xy_pulse;
    *out_z_pulse++ = z_pulse;

    previous_phase = new_phase;
  }

  x->x_x = x_x;
  x->x_y = x_y;
  x->x_z = x_z;
  x->x_previous_phase_sector = previous_phase;
  // x->x_predicted_z_amp = predicted_z_amp;

  return (w+9);
}

static void rossler_dsp(t_rossler *x, t_signal **sp) {
  x->x_conv = (t_float)((t_float)1.0 / sp[0]->s_sr);
  dsp_add(rossler_perform, 8, x,
          sp[0]->s_vec,
          sp[1]->s_vec,
          sp[2]->s_vec,
          sp[3]->s_vec,
          sp[4]->s_vec,
          sp[5]->s_vec,
          sp[0]->s_length);
}

static void reset(t_rossler *x) {
  x->x_x = x->x_x_init;
  x->x_y = x->x_y_init;
  x->x_z = x->x_z_init;
}

static void set_a(t_rossler *x, t_floatarg f) {
  x->x_a = f;
}

static void set_b(t_rossler *x, t_floatarg f) {
  x->x_b = f;
}

static void set_c(t_rossler *x, t_floatarg f) {
  x->x_c = f;
}

static void set_z_pulse_threshold(t_rossler *x, t_floatarg f) {
  x->x_z_pulse_threshold = f;
}

static void set_x_pulse_width(t_rossler *x, t_floatarg f) {
  x->x_phase_pulse_sectors = (int)f;
}

void rossler_tilde_setup(void) {
  rossler_class = class_new(gensym("rossler~"),
                            (t_newmethod)rossler_new,
                            (t_method)rossler_free,
                            sizeof(t_rossler),
                            CLASS_DEFAULT,
                            A_GIMME, 0);
  class_addmethod(rossler_class, (t_method)rossler_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(rossler_class, (t_method)reset, gensym("reset"), 0);
  class_addmethod(rossler_class, (t_method)set_a, gensym("a"), A_FLOAT, 0);
  class_addmethod(rossler_class, (t_method)set_b, gensym("b"), A_FLOAT, 0);
  class_addmethod(rossler_class, (t_method)set_c, gensym("c"), A_FLOAT, 0);
  class_addmethod(rossler_class, (t_method)set_z_pulse_threshold,
                  gensym("z_pulse_threshold"), A_FLOAT, 0);
  class_addmethod(rossler_class, (t_method)set_x_pulse_width,
                  gensym("pulse_width"), A_FLOAT, 0);
  CLASS_MAINSIGNALIN(rossler_class, t_rossler, x_f);
}
