#include "m_pd.h"
#include <math.h>

/**
 * notes:
 * - henon_map.md
 **/

typedef struct _henon {
  t_object x_obj;

  t_float x_x;
  t_float x_y;

  t_float x_init_x;
  t_float x_init_y;

  t_float x_fp1_x;
  t_float x_fp1_y;
  t_float x_fp2_x;
  t_float x_fp2_y;

  t_sample x_y1_pulse;
  t_sample x_y2_pulse;

  t_float x_a;
  t_float x_b;

  t_sample x_trig_prev;
  t_sample x_f;

  t_outlet *x_x_out;
  t_outlet *x_y_out;
  t_outlet *x_x_crossing_out;
  t_outlet *x_y1_crossing_out;
  t_outlet *x_y2_crossing_out;
} t_henon;

static t_class *henon_class = NULL;

static void henon_calculate_fixed_points(t_henon *x);

static void *henon_new(t_symbol *s, int argc, t_atom *argv) {
  t_henon *x = (t_henon *)pd_new(henon_class);

  t_float x_x, x_y;
  if (argc < 2) {
    x_x = (t_float)0.0;
    x_y = (t_float)0.0;
  } else {
    x_x = atom_getfloat(argv++);
    x_y = atom_getfloat(argv++);
  }

  x->x_x = x_x;
  x->x_init_x = x_x;
  x->x_y = x_y;
  x->x_init_y = x_y;

  x->x_a = (t_float)1.4;
  x->x_b = (t_float)0.3;

  // assigns values to fixt point attributes
  henon_calculate_fixed_points(x);

  x->x_y1_pulse = (t_sample)-1.0;
  x->x_y2_pulse = (t_sample)-1.0;

  x->x_trig_prev = (t_sample)0.0;
  x->x_f = (t_sample)1.0;

  x->x_y2_crossing_out = outlet_new(&x->x_obj, &s_signal);
  x->x_y1_crossing_out = outlet_new(&x->x_obj, &s_signal);
  x->x_x_crossing_out = outlet_new(&x->x_obj, &s_signal);
  x->x_y_out = outlet_new(&x->x_obj, &s_signal);
  x->x_x_out = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void henon_free(t_henon *x) {
  if (x->x_y_out) outlet_free(x->x_y_out);
  if (x->x_x_out) outlet_free(x->x_x_out);
}

static t_int *henon_perform(t_int *w) {
  t_henon *x = (t_henon *)(w[1]);
  t_sample *in_trig = (t_sample *)(w[2]);
  t_sample *out_x = (t_sample *)(w[3]);
  t_sample *out_y = (t_sample *)(w[4]);
  t_sample *out_x_crossing = (t_sample *)(w[5]);
  t_sample *out_y1_crossing = (t_sample *)(w[6]);
  t_sample *out_y2_crossing = (t_sample *)(w[7]);
  int n = (int)(w[8]);

  t_float x_x = x->x_x;
  t_float x_y = x->x_y;
  t_float a = x->x_a;
  t_float b = x->x_b;

  t_sample prev_trig = x->x_trig_prev;

  t_sample y1_pulse = x->x_y1_pulse; // initialized to -1
  t_sample y2_pulse = x->x_y2_pulse; // initialized to -1

  t_float fp1_y = x->x_fp1_y; // default 0.189
  t_float fp2_y = x->x_fp2_y; // default -0.265

  while (n--) {
    t_sample next_trig = *in_trig++;

    if (next_trig > prev_trig) {
      t_sample next_x = (t_float)1.0 - a * x_x*x_x + x_y;
      x_y = b * x_x;
      x_x = next_x;
      if (x_x >= 0) {
        y1_pulse = (x_y >= fp1_y) ? (t_sample)1.0 : (t_sample)-1.0;
        y2_pulse = (t_sample)-1.0;
      } else {
        y2_pulse = (x_y >= fp2_y) ? (t_sample)1.0 : (t_sample)-1.0;
        y1_pulse = (t_sample)-1.0;
      }
    }

    *out_x++ = x_x;
    *out_y++ = x_y;
    *out_x_crossing++ = (x_x >= 0) ? (t_sample)1.0 : (t_sample)-1.0;
    *out_y1_crossing++ = y1_pulse;
    *out_y2_crossing++ = y2_pulse;

    prev_trig = next_trig;
  }

  x->x_x = x_x;
  x->x_y = x_y;
  x->x_trig_prev = prev_trig;
  x->x_y1_pulse = y1_pulse;
  x->x_y2_pulse = y2_pulse;

  return (w+9);
}

static void henon_dsp(t_henon *x, t_signal **sp) {
  dsp_add(henon_perform, 8,
          x,
          sp[0]->s_vec, // trig inlet
          sp[1]->s_vec, // x out
          sp[2]->s_vec, // y out
          sp[3]->s_vec, // x-crossing out
          sp[4]->s_vec, // y1-crossing out
          sp[5]->s_vec, // y2-crossing out
          sp[0]->s_length);
}

static void henon_calculate_fixed_points(t_henon *x) {
  t_float a = x->x_a;
  t_float b = x->x_b;

  t_float discriminant = ((t_float)1-b)*((t_float)1-b) + (t_float)4*a;

  if (discriminant >= 0) {
    t_float sqrt_disc = sqrtf(discriminant);

    x->x_fp1_x = (-((t_float)1-b) + sqrt_disc) / ((t_float)2*a);
    x->x_fp1_y = b * x->x_fp1_x;

    x->x_fp2_x = (-((t_float)1-b) - sqrt_disc) / ((t_float)2*a);
    x->x_fp2_y = b * x->x_fp2_x;
  }
  post("x_fp1_y: %f, x_fp2_y: %f", x->x_fp1_y, x->x_fp2_y);
}

static void reset(t_henon *x) {
  x->x_x = x->x_init_x;
  x->x_y = x->x_init_y;
}

static void set_a(t_henon *x, t_floatarg f) {
  x->x_a = f;
  henon_calculate_fixed_points(x);
}

static void set_b(t_henon *x, t_floatarg f) {
  x->x_b = f;
  henon_calculate_fixed_points(x);
}

void henon_tilde_setup(void) {
  henon_class = class_new(gensym("henon~"),
                          (t_newmethod)henon_new,
                          (t_method)henon_free,
                          sizeof(t_henon),
                          CLASS_DEFAULT,
                          A_GIMME,
                          0);

  class_addmethod(henon_class, (t_method)henon_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(henon_class, (t_method)reset, gensym("reset"), 0);
  class_addmethod(henon_class, (t_method)set_a, gensym("a"), A_FLOAT, 0);
  class_addmethod(henon_class, (t_method)set_b, gensym("b"), A_FLOAT, 0);
  CLASS_MAINSIGNALIN(henon_class, t_henon, x_f);
}
