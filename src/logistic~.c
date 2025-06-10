#include "m_pd.h"
#include <math.h>

/**
* notes: logistic_map.md
**/

typedef struct _logistic {
  t_object x_obj;

  t_float x_x;
  t_float x_init_x;

  t_float x_theta;
  t_float x_init_theta;
  // t_float x_theta_prev;
  t_sample x_rotation_pulse;

  t_float x_r;

  t_float x_trig_prev;

  t_float x_f; // dummy arg

  t_outlet *x_theta_pulse_out;
  t_outlet *x_theta_out;
  t_outlet *x_x_out;
} t_logistic;

static t_class *logistic_class = NULL;

static void *logistic_new(t_symbol *s, int argc, t_atom *argv) {
  t_logistic *x = (t_logistic *)pd_new(logistic_class);

  t_float x_x, x_theta;
  if (argc < 2) { // this is a bad pattern!
    x_x = (t_float)0.01;
    x_theta = (t_float)0.0;
  } else {
    x_x = atom_getfloat(argv++);
    x_theta = atom_getfloat(argv++);
  }
  x->x_x = x_x;
  x->x_init_x = x_x;
  x->x_theta = x_theta;
  x->x_init_theta = x_theta;
  // x->x_theta_prev = (t_sample)1.0;
  x->x_rotation_pulse = (t_sample)1.0; // maybe should be -1.0?

  x->x_r = (t_float)3.57;

  x->x_trig_prev = (t_float)0.0;

  x->x_f = (t_float)1.0;

  x->x_theta_pulse_out = outlet_new(&x->x_obj, &s_signal);
  x->x_theta_out = outlet_new(&x->x_obj, &s_signal);
  x->x_x_out = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void logistic_free(t_logistic *x) {
  if (x->x_x_out) outlet_free(x->x_x_out);
  if (x->x_theta_out) outlet_free(x->x_theta_out);
  if (x->x_theta_pulse_out) outlet_free(x->x_theta_pulse_out);
}

static t_int *logistic_perform(t_int *w) {
  t_logistic *x = (t_logistic *)(w[1]);
  t_sample *in_trig = (t_sample *)(w[2]);
  t_sample *out_x = (t_sample *)(w[3]);
  t_sample *out_theta = (t_sample *)(w[4]);
  t_sample *out_rotation_pulse = (t_sample *)(w[5]);
  int n = (int)(w[6]);

  t_float x_x = x->x_x;
  t_float r = x->x_r;

  t_float theta = x->x_theta;

  t_float prev_trig = x->x_trig_prev;
  t_sample rotation_pulse = x->x_rotation_pulse;

  while (n--) {
    t_sample next_trig = *in_trig++;

    if (next_trig > prev_trig) {
      t_float prev = theta;
      theta = theta + x_x;
      theta = theta - floorf(theta);
      x_x = r * x_x * ((t_float)1.0 - x_x);
      if (theta < prev) {
        rotation_pulse = (t_sample)1.0;
      } else {
        rotation_pulse = (t_sample)-1.0;
      }
    }

    *out_x++ = x_x;
    *out_theta++ = theta;
    *out_rotation_pulse++ = rotation_pulse;
    prev_trig = next_trig;
  }

  x->x_x = x_x;
  x->x_trig_prev = prev_trig;
  x->x_theta = theta;
  x->x_rotation_pulse = rotation_pulse;

  return (w+7);
}

static void logistic_dsp(t_logistic *x, t_signal **sp) {
  dsp_add(logistic_perform, 6,
          x,
          sp[0]->s_vec,
          sp[1]->s_vec,
          sp[2]->s_vec,
          sp[3]->s_vec,
          sp[0]->s_length);
}

static void reset(t_logistic *x) {
  x->x_x = x->x_init_x;
  x->x_theta = x->x_init_theta;
}

static void set_r(t_logistic *x, t_floatarg f) {
  x->x_r = f;
}

void logistic_tilde_setup(void) {
  logistic_class = class_new(gensym("logistic~"),
                             (t_newmethod)logistic_new,
                             (t_method)logistic_free,
                             sizeof(t_logistic),
                             CLASS_DEFAULT,
                             A_GIMME, 0);

  class_addmethod(logistic_class, (t_method)logistic_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(logistic_class, (t_method)reset, gensym("reset"), 0);
  class_addmethod(logistic_class, (t_method)set_r, gensym("r"), A_FLOAT, 0);
  CLASS_MAINSIGNALIN(logistic_class, t_logistic, x_f);
}


