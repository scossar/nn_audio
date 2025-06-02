#include "m_pd.h"
#include <math.h>

/**
 * notes:
 * - lorenz_systems.md
 * - understanding_the_lorenz_equations.md
 **/

typedef struct _lorenzattr {
  t_object x_obj;

  t_float x_x;
  t_float x_y;
  t_float x_z;

  t_float x_init_x;
  t_float x_init_y;
  t_float x_init_z;

  t_float x_dt;

  t_float x_sigma;
  t_float x_rho;
  t_float x_beta;

  t_float x_2_recip;
  t_float x_6_recip;
  t_float x_wing_center_base;
  int x_previous_phase_sector;
  t_sample x_wing_phase_pulse;
  int x_phase_pulse_sectors;

  t_float x_conv;

  t_outlet *x_wing_phase_pulse_out; // alternates 1,0 on each phase section
  t_outlet *x_x_crossing_out;
  t_outlet *x_wing_phase_out; // outputs the new phase
  t_outlet *x_z_outlet;
  t_outlet *x_y_outlet;
  t_outlet *x_x_outlet; // when more than one signal outlet is needed, all
  // signal outlets need to be explicitly created
} t_lorenzattr;

static t_class *lorenzattr_class = NULL;

static void reset(t_lorenzattr *x);

static void *lorenzattr_new(t_symbol *s, int argc, t_atom *argv) {
  t_lorenzattr *x = (t_lorenzattr *)pd_new(lorenzattr_class);

  t_float x_x, x_y, x_z;

  if (argc < 3) {
    x_x = (t_float)1.0;
    x_y = (t_float)1.0;
    x_z = (t_float)1.0;
  } else {
    x_x = atom_getfloat(argv++);
    x_y = atom_getfloat(argv++);
    x_z = atom_getfloat(argv++);
  }

  x->x_x = x_x;
  x->x_init_x = x_x;
  x->x_y = x_y;
  x->x_init_y = x_y;
  x->x_z = x_z;
  x->x_init_z = x_z;

  // note: technically this should be x_f (look at how it gets used in the perform function)
  x->x_dt = (t_float)0.0001;
  x->x_conv = (t_float)0.0;

  x->x_sigma = (t_float)10.0;
  x->x_rho = (t_float)28.0;
  x->x_beta = (t_float)((t_float)8.0 / (t_float)3.0);

  x->x_2_recip = (t_float)0.5; // for completeness
  x->x_6_recip = (t_float)((t_float)1.0 / (t_float)6.0);
  x->x_wing_center_base = (t_float)(sqrtf(x->x_beta * (x->x_rho - (t_float)1.0)));
  x->x_previous_phase_sector = 0;
  x->x_phase_pulse_sectors = 1;
  x->x_wing_phase_pulse = (t_sample)1.0;

  x->x_wing_phase_pulse_out = outlet_new(&x->x_obj, &s_signal);
  x->x_x_crossing_out = outlet_new(&x->x_obj, &s_signal);
  x->x_wing_phase_out = outlet_new(&x->x_obj, &s_signal);
  x->x_z_outlet = outlet_new(&x->x_obj, &s_signal);
  x->x_y_outlet = outlet_new(&x->x_obj, &s_signal);
  x->x_x_outlet = outlet_new(&x->x_obj, &s_signal);

  return (void *)x;
}

static void lorenzattr_free(t_lorenzattr *x) {
  if (x->x_wing_phase_pulse_out) outlet_free(x->x_wing_phase_pulse_out);
  if (x->x_x_crossing_out) outlet_free(x->x_x_crossing_out);
  if (x->x_wing_phase_out) outlet_free(x->x_wing_phase_out);
  if (x->x_z_outlet) outlet_free(x->x_z_outlet);
  if (x->x_y_outlet) outlet_free(x->x_y_outlet);
  if (x->x_x_outlet) outlet_free(x->x_x_outlet);
}

static int get_lorenz_phase_sector(t_lorenzattr *x, t_float x_x, t_float x_y) {
  t_float base_center = x->x_wing_center_base;
  t_float center_xy = (x_x > (t_float)0.0) ? base_center : -base_center;

  t_float dx = x_x - center_xy;
  t_float dy = x_y - center_xy;

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

static t_int *lorenzattr_perform(t_int *w) {
  t_lorenzattr *x = (t_lorenzattr *)(w[1]);
  t_sample *in_dt = (t_sample *)(w[2]);
  t_sample *out_x = (t_sample *)(w[3]);
  t_sample *out_y = (t_sample *)(w[4]);
  t_sample *out_z = (t_sample *)(w[5]);
  t_sample *out_wing_phase = (t_sample *)(w[6]);
  t_sample *out_x_crossing = (t_sample *)(w[7]);
  t_sample *out_wing_pulse = (t_sample *)(w[8]);
  int n = (int)(w[9]);

  t_float x_x = x->x_x;
  t_float x_y = x->x_y;
  t_float x_z = x->x_z;

  t_float sigma = x->x_sigma;
  t_float rho = x->x_rho;
  t_float beta = x->x_beta;

  const t_float recip_2 = x->x_2_recip;
  const t_float recip_6 = x->x_6_recip;

  int new_phase = 0;
  int previous_phase = x->x_previous_phase_sector;
  int pulse_sectors = x->x_phase_pulse_sectors;
  t_sample wing_phase_pulse = x->x_wing_phase_pulse;

  t_float conv = x->x_conv;

  while (n--) {
    t_sample f = *in_dt++;
    t_float dt = f * conv;
    // it explodes at values around 0.1
    if (dt > (t_float)0.1) dt = (t_float)0.1;
    if (dt <= (t_float)0.0) dt = (t_float)1e-10;

    t_float k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z, k4x, k4y, k4z;
    t_float x_temp, y_temp, z_temp;

    k1x = sigma * (x_y - x_x);
    k1y = x_x * (rho - x_z) - x_y;
    k1z = x_x * x_y - beta * x_z;

    x_temp = x_x + dt * k1x * recip_2;
    y_temp = x_y + dt * k1y * recip_2;
    z_temp = x_z + dt * k1z * recip_2;

    k2x = sigma * (y_temp - x_temp);
    k2y = x_temp * (rho - z_temp) - y_temp;
    k2z = x_temp * y_temp - beta * z_temp;

    x_temp = x_x + dt * k2x * recip_2;
    y_temp = x_y + dt * k2y * recip_2;
    z_temp = x_z + dt * k2z * recip_2;

    k3x = sigma * (y_temp - x_temp);
    k3y = x_temp * (rho - z_temp) - y_temp;
    k3z = x_temp * y_temp - beta * z_temp;

    x_temp = x_x + dt * k3x;
    y_temp = x_y + dt * k3y;
    z_temp = x_z + dt * k3z;

    k4x = sigma * (y_temp - x_temp);
    k4y = x_temp * (rho - z_temp) - y_temp;
    k4z = x_temp * y_temp - beta * z_temp;

    x_x += dt * (k1x + 2*k2x + 2*k3x + k4x) * recip_6;
    x_y += dt * (k1y + 2*k2y + 2*k3y + k4y) * recip_6;
    x_z += dt * (k1z + 2*k2z + 2*k3z + k4z) * recip_6;

    new_phase = get_lorenz_phase_sector(x, x_x, x_y);
    t_sample wing_new_phase_pulse = (t_sample)0.0;
    if (new_phase < pulse_sectors) wing_new_phase_pulse = (t_sample)1.0;

    // feels like a hack, will fix
    if (previous_phase != new_phase) {
      wing_phase_pulse = (wing_phase_pulse == (t_sample)1.0) ? (t_sample)0.0 : (t_sample)1.0; 
    }

    // (x > 0) ? right wing : left wing
    t_sample x_crossing_pulse = x_x > (t_float)0.0 ? (t_sample)1.0 : (t_sample)0.0;

    previous_phase = new_phase;

    *out_x++ = x_x;
    *out_y++ = x_y;
    *out_z++ = x_z;
    *out_wing_phase++ = wing_new_phase_pulse;
    *out_x_crossing++ = x_crossing_pulse;
    *out_wing_pulse++ = wing_phase_pulse;
  }
  x->x_x = x_x;
  x->x_y = x_y;
  x->x_z = x_z;
  x->x_previous_phase_sector = previous_phase;
  x->x_wing_phase_pulse = wing_phase_pulse;

  return (w+10);
}

static void lorenzattr_dsp(t_lorenzattr *x, t_signal **sp) {
  x->x_conv = (t_float)1.0 / sp[0]->s_sr;
  dsp_add(lorenzattr_perform, 9,
          x,
          sp[0]->s_vec, // dt
          sp[1]->s_vec, // x_outlet
          sp[2]->s_vec, // y_outlet
          sp[3]->s_vec, // z_outlet
          sp[4]->s_vec, // wing_phase_out
          sp[5]->s_vec, // x_crossing_out
          sp[6]->s_vec, // wing_phase_pulse
          sp[0]->s_length);
}

static void reset(t_lorenzattr *x) {
  x->x_x = x->x_init_x;
  x->x_y = x->x_init_y;
  x->x_z = x->x_init_z;
}

static void set_sigma(t_lorenzattr *x, t_floatarg f) {
  x->x_sigma = f;
}

static void set_rho(t_lorenzattr *x, t_floatarg f) {
  x->x_rho = f;
}

static void set_beta(t_lorenzattr *x, t_floatarg f) {
  x->x_beta = f;
}

static void set_pulse_width(t_lorenzattr *x, t_floatarg f) {
  x->x_phase_pulse_sectors = (int)f;
}

void lorenzattr_tilde_setup(void) {
  lorenzattr_class = class_new(gensym("lorenzattr~"),
                           (t_newmethod)lorenzattr_new,
                           (t_method)lorenzattr_free,
                           sizeof(t_lorenzattr),
                           CLASS_DEFAULT,
                           A_GIMME,
                           0);

  class_addmethod(lorenzattr_class, (t_method)lorenzattr_dsp, gensym("dsp"), A_CANT, 0);
  class_addmethod(lorenzattr_class, (t_method)reset, gensym("reset"), 0);
  class_addmethod(lorenzattr_class, (t_method)set_sigma, gensym("sigma"), A_FLOAT, 0);
  class_addmethod(lorenzattr_class, (t_method)set_rho, gensym("rho"), A_FLOAT, 0);
  class_addmethod(lorenzattr_class, (t_method)set_beta, gensym("beta"), A_FLOAT, 0);
  class_addmethod(lorenzattr_class, (t_method)set_pulse_width, gensym("pulse_width"), A_FLOAT, 0);

  CLASS_MAINSIGNALIN(lorenzattr_class, t_lorenzattr, x_dt);
}
