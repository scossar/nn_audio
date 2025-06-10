#include "m_pd.h"
#include <math.h>

/**
* notes: logistic_map.md
**/

typedef struct _clogistic {
  t_object x_obj;

  t_float x_x;
  t_float x_init_x;

  t_float x_theta;
  t_float x_init_theta;

  t_float x_r;

  t_sample x_rotation_pulse;

  t_float x_coupling_strength;

  t_float x_prev_input;
  t_float x_f; // dummy arg

  t_inlet *x_mod_in;

  t_outlet *x_rotation_pulse_out;
  t_outlet *x_theta_out;
  t_outlet *x_x_out;
} t_clogistic;

static t_class *clogistic_class = NULL;

static void *clogistic_new(t_symbol *s, int argc, t_atom *argv) {
  t_clogistic *x = (t_clogistic *)pd_new(clogistic_class);

  t_float x_x, x_theta;
  // lazy pattern:
  if (argc < 2) {
    x_x = (t_float)0.1;
    x_theta = (t_float)0.0;
  } else {
    x_x = atom_getfloat(argv++);
    x_theta = atom_getfloat(argv++);
  }

  x->x_x = x_x;
  x->x_init_x = x_x;
  x->x_theta = x_theta;
  x->x_init_theta = x_theta;

  x->x_rotation_pulse = (t_sample)1.0; // maybe should be -1.0?
  
  x->x_r = (t_float)3.57;
  x->x_prev_input = (t_float)0.0;

  x->x_f = (t_float)0.0; // this works, but feels like a hack

  x->x_coupling_strength = (t_float)0.01;

  x->x_mod_in = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);

  x->x_rotation_pulse_out = outlet_new(&x->x_obj, &s_signal);
  x->x_theta_out = outlet_new(&x->x_obj, &s_signal);
  x->x_x_out = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void clogistic_free(t_clogistic *x) {
  if (x->x_rotation_pulse_out) outlet_free(x->x_rotation_pulse_out);
  if (x->x_theta_out) outlet_free(x->x_theta_out);
  if (x->x_x_out) outlet_free(x->x_x_out);
  if (x->x_mod_in) inlet_free(x->x_mod_in);
}

static t_int *clogistic_perform(t_int *w) {
  t_clogistic *x = (t_clogistic *)(w[1]);
  t_sample *trig_in = (t_sample *)(w[2]);
  t_sample *mod_in = (t_sample *)(w[3]);
  t_sample *x_out = (t_sample *)(w[4]);
  t_sample *theta_out = (t_sample *)(w[5]);
  t_sample *rotation_pulse_out = (t_sample *)(w[6]);
  int n = (int)(w[7]);

  t_float x_x = x->x_x;
  t_float r = x->x_r;
  t_float theta = x->x_theta;
  t_float prev_input = x->x_prev_input;
  t_sample rotation_pulse = x->x_rotation_pulse;
  t_float coupling_strength = x->x_coupling_strength;

  while (n--) {
    t_sample next_input = *trig_in++;
    t_sample mod = *mod_in++;

    if (next_input > prev_input) {
      t_float prev_theta = theta;
      theta = theta + x_x;
      theta = theta - floorf(theta);
      rotation_pulse = (theta < prev_theta) ? (t_sample)1.0 : (t_sample)-1.0;

      // r = r + coupling_strength * mod;
      x_x = r * x_x * ((t_float)1.0 - x_x + coupling_strength * mod);
    }

    *x_out++ = x_x;
    *theta_out++ = theta;
    *rotation_pulse_out++ = rotation_pulse;

    prev_input = next_input;
  }

  x->x_x = x_x;
  x->x_prev_input = prev_input;
  x->x_theta = theta;
  x->x_rotation_pulse = rotation_pulse;

  return (w+8);
}

static void clogistic_dsp(t_clogistic *x, t_signal **sp) {
  dsp_add(clogistic_perform, 7,
          x,
          sp[0]->s_vec,
          sp[1]->s_vec,
          sp[2]->s_vec,
          sp[3]->s_vec,
          sp[4]->s_vec,
          sp[0]->s_length);
}

static void reset(t_clogistic *x) {
  x->x_x = x->x_init_x;
  x->x_theta = x->x_init_theta;
}

static void set_r(t_clogistic *x, t_floatarg f) {
  x->x_r = f;
}

static void set_coupling_strength(t_clogistic *x, t_floatarg f) {
  x->x_coupling_strength = f;
}

void clogistic_tilde_setup(void) {
  clogistic_class = class_new(gensym("clogistic~"),
                             (t_newmethod)clogistic_new,
                             (t_method)clogistic_free,
                             sizeof(t_clogistic),
                             CLASS_DEFAULT,
                             A_GIMME, 0);

  class_addmethod(clogistic_class, (t_method)clogistic_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(clogistic_class, (t_method)reset, gensym("reset"), 0);
  class_addmethod(clogistic_class, (t_method)set_r, gensym("r"), A_FLOAT, 0);
  class_addmethod(clogistic_class, (t_method)set_coupling_strength, gensym("coupling"), A_FLOAT, 0);
  CLASS_MAINSIGNALIN(clogistic_class, t_clogistic, x_f);
}
