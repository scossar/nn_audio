#include "m_pd.h"
#include <math.h>

/**
* notes:
* - simple_chaotic_systems.md
**/

#define CLAMP(x, min, max) ((x) < (min) ? (min) : ((x) > (max) ? (max) : (x)))

typedef struct _tent {
  t_object x_obj;

  t_float x_x;
  t_float x_init_x;
  t_float x_prev_x;

  t_float x_theta;
  t_float x_init_theta;
  t_sample x_rotation_pulse;

  t_float x_r;
  t_float x_coupling_strength;

  t_float x_prev_trigger;
  t_float x_f; // dummy arg

  t_inlet *x_mod_in;

  t_outlet *x_rotation_pulse_out;
  t_outlet *x_theta_out;
  t_outlet *x_prev_out;
  t_outlet *x_x_out;
} t_tent;

static t_class *tent_class;

static void *tent_new(t_symbol *s, int argc, t_atom *argv) {
  t_tent *x = (t_tent *)pd_new(tent_class);

  t_float x_x;
  if (argc < 1) {
    x_x = (t_float)0.25;
  } else {
    x_x = atom_getfloat(argv++);
  }
  x->x_x = x_x;
  x->x_init_x = x_x;
  x->x_prev_x = x_x;

  x->x_theta = (t_float)0.0; // might make user configurable
  x->x_init_theta = x->x_theta;

  x->x_r = (t_float)1.5;
  x->x_coupling_strength = (t_float)0.01;

  x->x_prev_trigger = (t_float)0.0;
  x->x_rotation_pulse = (t_sample)1.0;

  x->x_f = (t_float)1.0;

  x->x_mod_in = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);

  x->x_rotation_pulse_out = outlet_new(&x->x_obj, &s_signal);
  x->x_theta_out = outlet_new(&x->x_obj, &s_signal);
  x->x_prev_out = outlet_new(&x->x_obj, &s_signal);
  x->x_x_out = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void tent_free(t_tent *x) {
  if (x->x_mod_in) inlet_free(x->x_mod_in);
  if (x->x_rotation_pulse_out) outlet_free(x->x_rotation_pulse_out);
  if (x->x_theta_out) outlet_free(x->x_theta_out);
  if (x->x_prev_out) outlet_free(x->x_prev_out);
  if (x->x_x_out) outlet_free(x->x_x_out);
}

static t_int *tent_perform(t_int *w) {
  t_tent *x = (t_tent *)(w[1]);
  t_sample *trig_in = (t_sample *)(w[2]);
  t_sample *mod_in = (t_sample *)(w[3]);
  t_sample *x_out = (t_sample *)(w[4]);
  t_sample *x_prev_x = (t_sample *)(w[5]);
  t_sample *theta_out = (t_sample *)(w[6]);
  t_sample *rotation_pulse_out = (t_sample *)(w[7]);
  int n = (int)(w[8]);

  t_float x_x = x->x_x;
  t_float r = x->x_r;
  t_float theta = x->x_theta;
  t_float prev_trigger = x->x_prev_trigger;
  t_sample rotation_pulse = x->x_rotation_pulse;
  t_sample prev_x = x->x_prev_x;
  t_float coupling_strength = x->x_coupling_strength;

  while (n--) {
    t_sample next_input = *trig_in++;
    t_sample mod = *mod_in++;

    if (next_input > prev_trigger) {
      t_float prev_theta = theta;
      theta += x_x;
      theta = theta - floorf(theta);
      rotation_pulse = (theta < prev_theta) ? (t_sample)1.0 : (t_sample)-1.0;

      prev_x = x_x;
      t_float x_next = (x_x < (t_float)0.5) ? x_x * r : r * ((t_float)1.0 - x_x);
      x_x = ((t_float)1.0 - coupling_strength) * x_next + coupling_strength * mod;
    }

    *x_out++ = x_x;
    *x_prev_x++ = prev_x;
    *theta_out++ = theta;
    *rotation_pulse_out++ = rotation_pulse;

    prev_trigger = next_input;
  }

  x->x_x = x_x;
  x->x_prev_x = prev_x;
  x->x_prev_trigger = prev_trigger;
  x->x_theta = theta;
  x->x_rotation_pulse = rotation_pulse;

  return (w+9);
}

static void tent_dsp(t_tent *x, t_signal **sp) {
  dsp_add(tent_perform, 8,
          x,
          sp[0]->s_vec, // trig in
          sp[1]->s_vec, // mod in
          sp[2]->s_vec, // x out
          sp[3]->s_vec, // x_prev out
          sp[4]->s_vec, // theta out
          sp[5]->s_vec, // rotation_pulse out
          sp[0]->s_length);
}

static void reset(t_tent *x) {
  x->x_x = x->x_init_x;
  x->x_theta = x->x_init_theta;
  x->x_prev_x = x->x_x;
}

static void set_r(t_tent *x, t_floatarg f) {
  x->x_r = f;
}

static void set_coupling_strength(t_tent *x, t_floatarg f) {
  x->x_coupling_strength = f;
}

void tent_tilde_setup(void) {
  tent_class = class_new(gensym("tent~"),
                         (t_newmethod)tent_new,
                         (t_method)tent_free,
                         sizeof(t_tent),
                         CLASS_DEFAULT,
                         A_GIMME, 0);

  class_addmethod(tent_class, (t_method)tent_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(tent_class, (t_method)reset, gensym("reset"), 0);
  class_addmethod(tent_class, (t_method)set_r, gensym("r"), A_FLOAT, 0);
  class_addmethod(tent_class, (t_method)set_coupling_strength, gensym("coupling"), A_FLOAT, 0);
  CLASS_MAINSIGNALIN(tent_class, t_tent, x_f);
}


