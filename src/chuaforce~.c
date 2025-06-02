#include "m_pd.h"

/**
 * notes:
 * - chua_circuit.md
 * - forcing_chua_and_other_chaotic_systems.md
 **/

typedef struct _chuaforce {
  t_object x_obj;

  t_float x_conv;
  t_float x_f;

  t_sample x_x;
  t_sample x_y;
  t_sample x_z;

  t_sample x_x_init;
  t_sample x_y_init;
  t_sample x_z_init;

  t_float x_alpha;
  t_float x_beta;
  t_float x_m0; // slope of the inner segment of the f(x) (chua's diode) function
  t_float x_m1; // slope of the outer segment of the f(x) function
  t_float x_bp; // breakpoint voltage

  t_float x_fixed_point_base;
  int x_current_xy_sector;
  int x_phase_pulse_sectors;
  t_sample x_x_pulse;

  t_float x_forcing_amp;

  t_float x_recip2;
  t_float x_recip6;

  t_inlet *x_forcing_frequency_in;

  t_outlet *x_x_transition_pulse_out;
  t_outlet *x_xy_pulse_out;
  t_outlet *x_z_out;
  t_outlet *x_y_out;
  t_outlet *x_x_out;
} t_chuaforce;

static t_class *chuaforce_class = NULL;

static void *chuaforce_new(t_symbol *s, int argc, t_atom *argv) {
  t_chuaforce *x = (t_chuaforce *) (t_chuaforce *)pd_new(chuaforce_class);

  t_sample x_x, x_y, x_z;

  if (argc < 3) {
    x_x = (t_sample)0.1;
    x_y = (t_sample)0.0;
    x_z = (t_sample)0.0;
  } else {
    x_x = (t_sample)(atom_getfloat(argv++));
    x_y = (t_sample)(atom_getfloat(argv++));
    x_z = (t_sample)(atom_getfloat(argv++));
}

  x->x_x = x_x;
  x->x_x_init = x_x;
  x->x_y = x_y;
  x->x_y_init = x_y;
  x->x_z = x_z;
  x->x_z_init = x_z;

  x->x_alpha = (t_float)15.6;
  x->x_beta = (t_float)28.58;
  x->x_m0 = (t_float)-1.143;
  x->x_m1 = (t_float)-0.714;
  x->x_bp = (t_float)1.0;

  x->x_fixed_point_base = x->x_bp * (x->x_m1 - x->x_m0) / (x->x_alpha * x->x_m1 + (t_float)1.0);
  x->x_current_xy_sector = 0;
  x->x_phase_pulse_sectors = 4;

  x->x_x_pulse = (t_sample)1.0;

  x->x_f = (t_float)1.0;
  x->x_conv = (t_float)0.0;
  x->x_recip2 = (t_float)0.5;
  x->x_recip6 = (t_float)1.0 / (t_float)6.0;

  x->x_forcing_amp = (t_float)0.01;

  x->x_forcing_frequency_in = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);

  x->x_x_transition_pulse_out = outlet_new(&x->x_obj, &s_signal);
  x->x_xy_pulse_out = outlet_new(&x->x_obj, &s_signal);
  x->x_z_out = outlet_new(&x->x_obj, &s_signal);
  x->x_y_out = outlet_new(&x->x_obj, &s_signal);
  x->x_x_out = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void chuaforce_free(t_chuaforce *x) {
  if (x->x_forcing_frequency_in) inlet_free(x->x_forcing_frequency_in);
  if (x->x_x_transition_pulse_out) outlet_free(x->x_x_transition_pulse_out);
  if (x->x_xy_pulse_out) outlet_free(x->x_xy_pulse_out);
  if (x->x_z_out) outlet_free(x->x_z_out);
  if (x->x_y_out) outlet_free(x->x_y_out);
  if (x->x_x_out) outlet_free(x->x_x_out);
}

static void chuaforce_derivatives(t_chuaforce *x,
                                  t_sample x_x, t_sample x_y, t_sample x_z, t_sample current_force,
                                  t_float *dx, t_float *dy, t_float *dz) {
  t_float alpha = x->x_alpha;
  t_float beta = x->x_beta;
  t_float m0 = x->x_m0;
  t_float m1 = x->x_m1;
  t_float bp = x->x_bp;
  t_float amp = x->x_forcing_amp;
  t_float forcing = current_force * amp;
  t_float f_x;

  if (x_x < -bp) {
      f_x = m1 * x_x - (m0 - m1) * bp;
  } else if (x_x > bp) {
      f_x = m1 * x_x + (m0 - m1) * bp;
  } else {
      f_x = m0 * x_x;
  }

  *dx = alpha * (x_y - x_x - f_x + forcing);
  *dy = x_x - x_y + x_z;
  *dz = -beta * x_y;
}

static int get_chuaforce_phase_sector(t_chuaforce *x, t_float x_x, t_float x_y) {
  t_float center_x = (x_x > 0) ? x->x_fixed_point_base : -x->x_fixed_point_base;
  t_float center_y = 0.0;

  t_float dx = x_x - center_x;
  t_float dy = x_y - center_y;

  int sector = 0;
  if (dy >= 0) {
    if (dx >= 0) {
      sector = (dy > dx) ? 1 : 0;
    } else {
      sector = (-dx > dy) ? 3 : 2;
    }
  } else {
    if (dx < 0) {
      sector = (-dy > -dx) ? 5 : 4;
    } else {
      sector = (dx > -dy) ? 7 : 6;
    }
  }

  return sector;
}

static t_int *chuaforce_perform(t_int *w) {
  t_chuaforce *x = (t_chuaforce *)(w[1]);
  t_sample *in_f = (t_sample *)(w[2]);
  t_sample *in_forcing = (t_sample *)(w[3]);
  t_sample *out_x = (t_sample *)(w[4]);
  t_sample *out_y = (t_sample *)(w[5]);
  t_sample *out_z = (t_sample *)(w[6]);
  t_sample *out_xy_pulse = (t_sample *)(w[7]);
  t_sample *out_x_transition = (t_sample *)(w[8]);
  int n = (int)(w[9]);

  t_float conv = x->x_conv; // 1. / sample_rate
  t_float recip2 = x->x_recip2; // 0.5
  t_float recip6 = x->x_recip6; // 1. / 6.

  t_sample x_x = x->x_x;
  t_sample x_y = x->x_y;
  t_sample x_z = x->x_z;

  t_sample x_pulse = x->x_x_pulse;

  int previous_sector = x->x_current_xy_sector; // might not get used
  int pulse_sectors = x->x_phase_pulse_sectors;

  while (n--) {
    t_sample f = *in_f++;
    t_float dt = f * conv;

    t_sample force = *in_forcing++;

    // this means the max value accepted from the frequency inlet (at s_r =
    // 48000) is 480
    // just a guess at this point. maybe it can go faster
    if (dt > (t_float)0.01) dt = (t_float)0.01;
    if (dt <= 0) dt = (t_float)1e-9;

    t_float k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
    t_float x_temp, y_temp, z_temp;
    t_float dt_half = dt * recip2;

    chuaforce_derivatives(x, x_x, x_y, x_z, force, &k1x, &k1y, &k1z);
    x_temp = x_x + k1x * dt_half;
    y_temp = x_y + k1y * dt_half;
    z_temp = x_z + k1z * dt_half;

    chuaforce_derivatives(x, x_temp, y_temp, z_temp, force, &k2x, &k2y, &k2z);
    x_temp = x_x + k2x * dt_half;
    y_temp = x_y + k2y * dt_half;
    z_temp = x_z + k2z * dt_half;

    chuaforce_derivatives(x, x_temp, y_temp, z_temp, force, &k3x, &k3y, &k3z);
    x_temp = x_x + k3x * dt;
    y_temp = x_y + k3y * dt;
    z_temp = x_z + k3z * dt;

    chuaforce_derivatives(x, x_temp, y_temp, z_temp, force, &k4x, &k4y, &k4z);

    x_x += dt * (k1x + (t_float)2.0*k2x + (t_float)2.0*k3x + k4x) * recip6;
    x_y += dt * (k1y + (t_float)2.0*k2y + (t_float)2.0*k3y + k4y) * recip6;
    x_z += dt * (k1z + (t_float)2.0*k2z + (t_float)2.0*k3z + k4z) * recip6;

    // get the current sector (0,7) of the x,y rotation around a fixed point
    int new_sector = get_chuaforce_phase_sector(x, x_x, x_y);

    // generates a pulse wave based on the phase of x,y
    t_sample xy_phase_pulse_out = (new_sector < pulse_sectors) ? (t_sample)1.0 : (t_sample)-1.0;

    // use the k4x derivative to determing if x is going to push the system to
    // the opposite fixed point
    if (x_x > (t_sample)0.0 && x_pulse == (t_sample)-1.0 && k4x > (t_float)0.0) {
      x_pulse = (t_sample)1.0;
    } else if (x_x < (t_sample)0.0 && x_pulse == (t_sample)1.0 && k4x < (t_float)0.0) {
      x_pulse = (t_sample)-1.0;
    }

    *out_x++ = x_x;
    *out_y++ = x_y;
    *out_z++ = x_z;
    *out_xy_pulse++ = xy_phase_pulse_out;
    *out_x_transition++ = x_pulse;

    // not currently used, but may get used to generate an pulse wave based on
    // alternating phase sectors
    previous_sector = new_sector; // not currently being used
  }

  x->x_x = x_x;
  x->x_y = x_y;
  x->x_z = x_z;
  x->x_current_xy_sector = previous_sector; // not yet used
  x->x_x_pulse = x_pulse;

  return (w+10);
}

static void chuaforce_dsp(t_chuaforce *x, t_signal **sp) {
  x->x_conv = (t_float)((t_float)1.0 / sp[0]->s_sr);
  dsp_add(chuaforce_perform, 9,
          x,
          sp[0]->s_vec,
          sp[1]->s_vec,
          sp[2]->s_vec,
          sp[3]->s_vec,
          sp[4]->s_vec,
          sp[5]->s_vec,
          sp[6]->s_vec,
          sp[0]->s_length);
}

static void reset(t_chuaforce *x) {
  x->x_x = x->x_x_init;
  x->x_y = x->x_y_init;
  x->x_z = x->x_z_init;
}

static void set_alpha(t_chuaforce *x, t_floatarg f) {
  x->x_alpha = f;
}

static void set_beta(t_chuaforce *x, t_floatarg f) {
  x->x_beta = f;
}

static void set_m0(t_chuaforce *x, t_floatarg f) {
  x->x_m0 = f;
}

static void set_m1(t_chuaforce *x, t_floatarg f) {
  x->x_m1 = f;
}

static void set_bp(t_chuaforce *x, t_floatarg f) {
  x->x_bp = f;
}

static void set_pulse_width(t_chuaforce *x, t_floatarg f) {
  x->x_phase_pulse_sectors = (int)f;
}

static void set_forcing_amp(t_chuaforce *x, t_floatarg f) {
  x->x_forcing_amp = f;
}

void chuaforce_tilde_setup(void) {
  chuaforce_class = class_new(gensym("chuaforce~"),
                         (t_newmethod)chuaforce_new,
                         (t_method)chuaforce_free,
                         sizeof(t_chuaforce),
                         CLASS_DEFAULT,
                         A_GIMME, 0);
  class_addmethod(chuaforce_class, (t_method)chuaforce_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(chuaforce_class, (t_method)reset, gensym("reset"), 0);
  class_addmethod(chuaforce_class, (t_method)set_pulse_width, gensym("pulse_width"), A_FLOAT, 0);
  class_addmethod(chuaforce_class, (t_method)set_forcing_amp, gensym("forcing_amp"), A_FLOAT, 0);
  class_addmethod(chuaforce_class, (t_method)set_alpha, gensym("alpha"), A_FLOAT, 0);
  class_addmethod(chuaforce_class, (t_method)set_beta, gensym("beta"), A_FLOAT, 0);
  class_addmethod(chuaforce_class, (t_method)set_m0, gensym("m0"), A_FLOAT, 0);
  class_addmethod(chuaforce_class, (t_method)set_m1, gensym("m1"), A_FLOAT, 0);
  class_addmethod(chuaforce_class, (t_method)set_bp, gensym("bp"), A_FLOAT, 0);
  CLASS_MAINSIGNALIN(chuaforce_class, t_chuaforce, x_f);
}
