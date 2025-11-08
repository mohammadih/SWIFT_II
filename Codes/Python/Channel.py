# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 01:17:17 2025

@author: Hossein
"""


import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper
from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import CDL
from sionna.phy.channel.tr38901.antenna import Antenna

# ----- Grid -----
fft_size = 1028
scs      = 30e3
n_sym    = 14
n_active = 1024          # actives you want to occupy
dc_null  = True
n_dc     = 1 if dc_null else 0
n_guard_total = fft_size - n_active - n_dc
assert n_guard_total >= 0
# allow asymmetric guards if needed
num_guard_carriers = (n_guard_total//2, n_guard_total - n_guard_total//2)
cp_len  = 72           # integer CP (samples)

RG = ResourceGrid(num_ofdm_symbols     = n_sym,
                  fft_size             = fft_size,
                  subcarrier_spacing   = scs,
                  num_tx               = 1,
                  num_streams_per_tx   = 1,
                  cyclic_prefix_length = cp_len,
                  num_guard_carriers   = num_guard_carriers,
                  dc_null              = dc_null,
                  pilot_pattern        = "empty")

# ----- Channel (CDL-D, SISO) -----
fc = 3.5e9
bs = Antenna(polarization="single", polarization_type="V",
             antenna_pattern="omni", carrier_frequency=fc)
ut = Antenna(polarization="single", polarization_type="V",
             antenna_pattern="omni", carrier_frequency=fc)

cdl = CDL(model="D",
          delay_spread=30e-9,
          carrier_frequency=fc,
          ut_array=ut,
          bs_array=bs,
          direction="uplink",
          min_speed=30.0, max_speed=30.0)

channel = OFDMChannel(channel_model=cdl,
                      resource_grid=RG,
                      normalize_channel=False,
                      return_channel=True)

# ----- Probe TF channel on the FULL grid -----
mapper = ResourceGridMapper(RG)
B = 1
num_data = RG.num_effective_subcarriers * n_sym  # active RE count
# all-ones on every DATA RE (guards/DC remain zero after mapping)
x_data = tf.complex(tf.ones([B, 1, 1, num_data]),
                    tf.zeros([B, 1, 1, num_data]))
x_rg = mapper(x_data)  # shape [B,Tx,Str,Sym,FFT] with zeros on nulled SCs

# FEED THE FULL GRID to the channel
y_rg, H_freq = channel(x_rg)  # H_freq: [B,Rx,RxAnt,Tx,TxAnt,Sym,FFT]

# ----- Build an "active mask" from the mapped grid and slice H on actives -----
# Reduce x_rg to a boolean mask [Sym, FFT]: True where data symbols exist
mask = tf.math.not_equal(tf.abs(tf.squeeze(x_rg, axis=[0,1,2])), 0.0)  # [Sym, FFT]
# Collapse Sym dimension to keep only subcarrier positions that are ever active
active_cols = tf.where(tf.reduce_any(mask, axis=0))[:, 0]              # [n_active]

# Extract |H| on active subcarriers
H_mag_full = tf.abs(tf.squeeze(H_freq, axis=[0,1,2,3,4]))  # [Sym, FFT]
H_mag_act  = tf.gather(H_mag_full, indices=active_cols, axis=1)  # [Sym, n_active]
H_mag_act  = tf.transpose(H_mag_act)  # [n_active, Sym] for plotting

plt.figure(figsize=(8,4))
plt.imshow(H_mag_act.numpy(), aspect='auto', origin='lower',
           extent=[0, n_sym-1, 0, int(active_cols.shape[0])-1])
plt.xlabel('OFDM symbol'); plt.ylabel('Active subcarrier index')
plt.title('‖H(f,t)‖ on active subcarriers'); plt.colorbar(); plt.tight_layout(); plt.show()
