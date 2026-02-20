"""
Session Viewer

Interactive viewer for a single photometry session: raw bands, preprocessed
signal, and peri-event heatmap + mean±SEM traces. All buttons are in a single
row at the top: region selectors on the left, Baseline/Mask toggles on the
right. Response panels are arranged as columns (one per event). Event onsets
are drawn as thin vertical lines in both signal traces.

Usage:
    python scripts/view_session.py [eid]
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import sem as scipy_sem
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

from iblnm.config import (
    SESSIONS_FPATH, SESSIONS_H5_DIR,
    QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH,
    SESSION_TYPES_TO_ANALYZE,
)
from iblnm.data import PhotometrySession
from iblnm.io import _get_default_connection
from iblnm.util import collect_session_errors

DEMO_EID = '2025366a-c9aa-4b6c-97be-8af40eda6410'

# Colours
COLOR_GCAMP = '#2ca02c'
COLOR_ISOSBESTIC = '#7f7f7f'
COLOR_PREPROCESSED = 'black'
EVENT_COLORS = {
    'stimOn_times':        '#1f77b4',   # blue
    'firstMovement_times': '#ff7f0e',   # orange
    'feedback_times':      '#2ca02c',   # green
}
EVENT_ORDER = ['stimOn_times', 'firstMovement_times', 'feedback_times']

# GridSpec height ratios
RATIO_SIGNAL   = 2
RATIO_HEATMAP  = 4
RATIO_MEAN     = 2


class PhotometrySessionViewer:

    def __init__(self, session):
        self.session = session
        self.fig = None
        self._axes = {}
        self._buttons = {}
        self._current_region = None
        self._baseline_on = False
        self._mask_on = False
        self._events_on = False

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _regions(self):
        return list(self.session.photometry['GCaMP_preprocessed'].columns)

    def _has_raw(self):
        return ('GCaMP' in self.session.photometry
                and 'Isosbestic' in self.session.photometry)

    def _has_responses(self):
        return (hasattr(self.session, 'responses')
                and self.session.responses is not None)

    def _events(self):
        """Return events sorted by EVENT_ORDER; unknowns appended at the end."""
        evs = list(self.session.responses.coords['event'].values)
        known   = [e for e in EVENT_ORDER if e in evs]
        unknown = [e for e in evs if e not in EVENT_ORDER]
        return known + unknown

    def _row_spec(self):
        """Return (row_types, height_ratios) for GridSpec construction.

        Signal rows span all event columns. Response rows have one column per
        event (heatmap row, then mean row).
        """
        types, ratios = [], []
        if self._has_raw():
            types.append('raw');       ratios.append(RATIO_SIGNAL)
        types.append('preprocessed'); ratios.append(RATIO_SIGNAL)
        if self._has_responses():
            types.append('heatmap');   ratios.append(RATIO_HEATMAP)
            types.append('mean');      ratios.append(RATIO_MEAN)
        return types, ratios

    # ------------------------------------------------------------------ #
    # Figure construction                                                  #
    # ------------------------------------------------------------------ #

    def _build_axes(self, fig, row_types, row_ratios):
        events = self._events() if self._has_responses() else []
        n_cols = max(len(events), 1)

        gs = GridSpec(
            len(row_types), n_cols, figure=fig,
            left=0.07, right=0.86, top=0.86, bottom=0.06,
            hspace=0.5, wspace=0.35, height_ratios=row_ratios,
        )
        axes = {}
        ref_signal = None   # first signal ax — others share its x

        for i, rt in enumerate(row_types):
            if rt == 'raw':
                ax = fig.add_subplot(gs[i, :])
                axes['raw'] = ax
                ref_signal = ax

            elif rt == 'preprocessed':
                kw = {'sharex': ref_signal} if ref_signal else {}
                ax = fig.add_subplot(gs[i, :], **kw)
                axes['preprocessed'] = ax
                if ref_signal is None:
                    ref_signal = ax

            elif rt == 'heatmap':
                for j, event in enumerate(events):
                    ax = fig.add_subplot(gs[i, j])
                    axes.setdefault('responses', {})[event] = {'heat': ax}

            elif rt == 'mean':
                for j, event in enumerate(events):
                    ax_heat = axes['responses'][event]['heat']
                    axes['responses'][event]['mean'] = fig.add_subplot(
                        gs[i, j], sharex=ax_heat
                    )

        return axes

    def _make_all_buttons(self, fig, regions):
        """Single row: region selectors (left group) + control toggles (right group)."""
        btn_w   = 0.10
        btn_h   = 0.04
        btn_y   = 0.90
        gap     = 0.012
        grp_gap = 0.030   # wider gap between region and control groups

        controls = [('events', 'Events')]
        if self._has_responses():
            controls += [('baseline', 'Baseline'), ('mask', 'Mask events')]

        region_w = len(regions) * btn_w + max(len(regions) - 1, 0) * gap
        ctrl_w   = (len(controls) * btn_w + max(len(controls) - 1, 0) * gap
                    if controls else 0)
        total_w  = region_w + (grp_gap + ctrl_w if controls else 0)
        x0 = 0.5 - total_w / 2

        for i, region in enumerate(regions):
            x = x0 + i * (btn_w + gap)
            ax_btn = fig.add_axes([x, btn_y, btn_w, btn_h])
            btn = Button(ax_btn, region, color='#e8e8e8', hovercolor='#a0c4ff')
            btn.on_clicked(lambda _, r=region: self._switch_region(r))
            self._buttons[region] = btn

        if controls:
            x_ctrl0 = x0 + region_w + grp_gap
            for i, (key, label) in enumerate(controls):
                x = x_ctrl0 + i * (btn_w + gap)
                ax_btn = fig.add_axes([x, btn_y, btn_w, btn_h])
                btn = Button(ax_btn, label, color='#e8e8e8', hovercolor='#a0c4ff')
                if key == 'events':
                    btn.on_clicked(self._toggle_events)
                elif key == 'baseline':
                    btn.on_clicked(self._toggle_baseline)
                else:
                    btn.on_clicked(self._toggle_mask)
                self._buttons[key] = btn

    def _add_event_lines(self, ax):
        """Draw per-trial event onset lines; return legend proxy handles."""
        from matplotlib.lines import Line2D
        if not self._events_on:
            return []
        if not hasattr(self.session, 'trials') or self.session.trials is None:
            return []
        handles = []
        xform = ax.get_xaxis_transform()
        for event, color in EVENT_COLORS.items():
            if event not in self.session.trials.columns:
                continue
            times = self.session.trials[event].dropna().values
            if len(times) == 0:
                continue
            ax.vlines(times, 0, 1, transform=xform, color=color, lw=1.0)
            handles.append(Line2D([0], [0], color=color, lw=1,
                                  label=event.replace('_times', '')))
        return handles

    # ------------------------------------------------------------------ #
    # Transforms                                                           #
    # ------------------------------------------------------------------ #

    def _toggle_events(self, _):
        self._events_on = not self._events_on
        color = '#a0c4ff' if self._events_on else '#e8e8e8'
        self._buttons['events'].ax.set_facecolor(color)
        self._switch_region(self._current_region)

    def _toggle_baseline(self, _):
        self._baseline_on = not self._baseline_on
        color = '#a0c4ff' if self._baseline_on else '#e8e8e8'
        self._buttons['baseline'].ax.set_facecolor(color)
        self._switch_region(self._current_region)

    def _toggle_mask(self, _):
        self._mask_on = not self._mask_on
        color = '#a0c4ff' if self._mask_on else '#e8e8e8'
        self._buttons['mask'].ax.set_facecolor(color)
        self._switch_region(self._current_region)

    def _get_transformed_responses(self):
        """Return responses with active transforms applied (for mean traces)."""
        resp = self.session.responses
        if self._mask_on:
            resp = self.session.mask_subsequent_events(resp)
        if self._baseline_on:
            resp = self.session.subtract_baseline(resp)
        return resp

    # ------------------------------------------------------------------ #
    # Drawing                                                              #
    # ------------------------------------------------------------------ #

    def _switch_region(self, region):
        self._current_region = region
        self._draw_region(region)
        self.fig.canvas.draw_idle()

    def _draw_region(self, region):
        if 'raw' in self._axes:
            self._plot_raw(self._axes['raw'], region)
        self._plot_preprocessed(self._axes['preprocessed'], region)
        for event, event_axes in self._axes.get('responses', {}).items():
            self._plot_responses(
                event_axes['heat'], event_axes['mean'], event, region
            )

    def _plot_raw(self, ax, region):
        from matplotlib.lines import Line2D
        ax.cla()
        gcamp = self.session.photometry['GCaMP'][region]
        iso   = self.session.photometry['Isosbestic'][region]
        t = gcamp.index.values
        ax.plot(t, gcamp.values, color=COLOR_GCAMP,      lw=1.2)
        ax.plot(t, iso.values,   color=COLOR_ISOSBESTIC, lw=1.2)
        signal_handles = [
            Line2D([0], [0], color=COLOR_GCAMP,      lw=1, label='GCaMP'),
            Line2D([0], [0], color=COLOR_ISOSBESTIC, lw=1, label='Isosbestic'),
        ]
        event_handles = self._add_event_lines(ax)
        ax.legend(handles=signal_handles + event_handles,
                  loc='upper left', bbox_to_anchor=(1.02, 1),
                  borderaxespad=0, fontsize=6)
        ax.set_ylabel('Signal (a.u.)', fontsize=7)
        ax.set_title(f'Raw — {region}', fontsize=8)
        ax.set_xlim(t[0], t[-1])
        ax.tick_params(labelbottom=False)

    def _plot_preprocessed(self, ax, region):
        ax.cla()
        signal = self.session.photometry['GCaMP_preprocessed'][region]
        t = signal.index.values
        ax.plot(t, signal.values, color=COLOR_PREPROCESSED, lw=1.2)
        ax.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.5)
        event_handles = self._add_event_lines(ax)
        if event_handles:
            ax.legend(handles=event_handles,
                      loc='upper left', bbox_to_anchor=(1.02, 1),
                      borderaxespad=0, fontsize=6)
        ax.set_ylabel('z-score', fontsize=7)
        ax.set_xlabel('Time (s)', fontsize=7)
        ax.set_title(f'Preprocessed — {region}', fontsize=8)
        ax.set_xlim(t[0], t[-1])

    def _plot_responses(self, ax_heat, ax_mean, event, region):
        ax_heat.cla()
        ax_mean.cla()

        tpts = self.session.responses.coords['time'].values
        raw  = self.session.responses.sel(region=region, event=event).values  # (n_trials, n_times)
        resp = self._get_transformed_responses().sel(region=region, event=event).values

        # Heatmap — always raw
        vmax = max(3 * np.nanstd(raw), 1e-6)
        ax_heat.imshow(
            raw, aspect='auto', origin='lower', cmap='RdBu_r',
            vmin=-vmax, vmax=vmax,
            extent=[tpts[0], tpts[-1], 0, raw.shape[0]],
        )
        ax_heat.axvline(0, color='k', lw=0.7, ls='--')
        ax_heat.set_ylabel('Trial', fontsize=7)
        ax_heat.set_title(event.replace('_times', ''), fontsize=8)
        ax_heat.tick_params(labelbottom=False)

        # Mean ± SEM — transformed
        mean = np.nanmean(resp, axis=0)
        err  = scipy_sem(resp, axis=0, nan_policy='omit')
        color = EVENT_COLORS.get(event, 'black')
        ax_mean.plot(tpts, mean, color=color, lw=1)
        ax_mean.fill_between(tpts, mean - err, mean + err, color=color, alpha=0.3)
        ax_mean.axhline(0, color='gray', ls='--', lw=0.5, alpha=0.5)
        ax_mean.axvline(0, color='gray', ls='--', lw=0.5, alpha=0.5)
        ax_mean.set_ylabel('z-score', fontsize=7)
        ax_mean.set_xlabel('Time (s)', fontsize=7)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def plot(self):
        regions = self._regions()
        self._current_region = regions[0]

        row_types, row_ratios = self._row_spec()
        events = self._events() if self._has_responses() else []
        fig_w = max(10, len(events) * 4)
        fig_h = max(9, len(row_types) * 2.5)
        fig = plt.figure(figsize=(fig_w, fig_h))
        self.fig = fig

        self._axes = self._build_axes(fig, row_types, row_ratios)
        self._make_all_buttons(fig, regions)

        s = self.session
        fig.suptitle(
            f"{s.subject} | {s.date} | {s.session_type} | {s.eid[:8]}",
            fontsize=10, y=0.99,
        )
        self._draw_region(self._current_region)
        return fig


# ======================================================================== #
# Main                                                                      #
# ======================================================================== #

if __name__ == '__main__':
    eid = sys.argv[1] if len(sys.argv) > 1 else DEMO_EID

    df = pd.read_parquet(SESSIONS_FPATH)
    df = collect_session_errors(df, [QUERY_DATABASE_LOG_FPATH, PHOTOMETRY_LOG_FPATH])
    fatal_errors = {'InvalidNeuromodulator', 'InvalidTarget', 'InvalidTargetNM'}
    df = df[df['logged_errors'].apply(
        lambda errs: not any(e in fatal_errors for e in errs)
    )]
    row = df[df['eid'] == eid].iloc[0]

    one = _get_default_connection()
    ps  = PhotometrySession(row, one=one)

    h5_path = SESSIONS_H5_DIR / f'{eid}.h5'
    if h5_path.exists():
        # Load raw bands first (load_photometry overwrites photometry dict),
        # then layer preprocessed + trials + responses from H5 on top.
        try:
            ps.load_photometry()
        except Exception as e:
            print(f"Warning: could not load raw photometry — {e}")
        ps.load_h5(h5_path)
    else:
        print(f"No H5 for {eid}, running pipeline...")
        ps.load_trials()
        ps.load_photometry()
        ps.preprocess()
        ps.extract_responses()
        ps.save_h5()

    viewer = PhotometrySessionViewer(ps)
    viewer.plot()
    plt.show()
