"""Campaign scheduling: geometry, gate census, grouping, order."""
from __future__ import annotations

import json

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from sedphot.measure import recipe
from sedphot.schedule import (
    DEFAULT_LINK_MARGIN_AS,
    PASS_HARVEST,
    PASS_PARALLEL,
    audit_groups,
    build_plan,
    census_from_catalog,
    connected_groups,
    consumer_counts,
    gate_reach_arcsec,
    link_radius_arcsec,
    read_targets,
    scene_radius_arcsec,
    write_plan,
)

NANOMAGGY_TO_UJY = 3631e6 / 1e9  # matches components.NANOMAGGY_TO_UJY
RA0, DEC0 = 210.0, 30.0


def _row(ra, dec, *, flux_ujy=500.0, rchisq=10.0, kind='SER'):
    """One Tractor row, in the columns the gate reads."""
    flux = flux_ujy / NANOMAGGY_TO_UJY
    row = {'ra': ra, 'dec': dec, 'type': kind}
    for band in ('g', 'r', 'i', 'z'):
        row[f'flux_{band}'] = flux
        row[f'rchisq_{band}'] = rchisq
    return row


def _offset(ra, dec, east_arcsec):
    """A position due east by a separation, in degrees."""
    return ra + east_arcsec / 3600.0 / np.cos(np.radians(dec)), dec


# ------------------------------------
# Geometry
# ------------------------------------

def test_radii_derive_from_the_stamp() -> None:
    # At the default stamp the recipe floor wins the cone, and the gate
    # reach is the half width less the edge margin.
    assert scene_radius_arcsec(120.0) == recipe.QUERY_RADIUS_AS
    assert gate_reach_arcsec(120.0) == 60.0 - recipe.GATE_EDGE_MARGIN_AS

    # A larger stamp grows both, so a hardcoded link distance would be
    # wrong exactly when the sweep is configured off the default.
    assert scene_radius_arcsec(400.0) > scene_radius_arcsec(120.0)
    assert gate_reach_arcsec(400.0) > gate_reach_arcsec(120.0)
    assert link_radius_arcsec(400.0) > link_radius_arcsec(120.0)


def test_gate_reach_stays_inside_the_stamp() -> None:
    # The census is radial only because the reach never escapes the
    # stamp's inscribed circle; if that stopped holding it would need
    # square geometry.
    for cutout in (60.0, 120.0, 240.0, 400.0):
        assert gate_reach_arcsec(cutout) < cutout / 2.0


def test_link_radius_covers_the_write_plus_read_span() -> None:
    link = link_radius_arcsec(120.0, margin_arcsec=0.0)
    assert link == gate_reach_arcsec(120.0) + scene_radius_arcsec(120.0)
    assert link_radius_arcsec(120.0) == link + DEFAULT_LINK_MARGIN_AS


# ------------------------------------
# Grouping and consumers
# ------------------------------------

def test_groups_split_at_the_link_radius() -> None:
    seps = [0.0, 100.0, 400.0, 460.0]
    coords = SkyCoord([_offset(RA0, DEC0, s)[0] for s in seps] * u.deg,
                      [DEC0] * len(seps) * u.deg)

    # 0-100 linked, 400-460 linked, the 300 gap between them is not.
    assert connected_groups(coords, 150.0) == [0, 0, 1, 1]
    # A radius past every gap collapses to one group.
    assert connected_groups(coords, 500.0) == [0, 0, 0, 0]
    # A radius under every gap isolates each.
    assert connected_groups(coords, 50.0) == [0, 1, 2, 3]


def test_group_labels_do_not_depend_on_input_order() -> None:
    seps = [0.0, 50.0, 900.0]
    coords = SkyCoord([_offset(RA0, DEC0, s)[0] for s in seps] * u.deg,
                      [DEC0] * 3 * u.deg)
    assert connected_groups(coords, 150.0) == [0, 0, 1]

    reversed_coords = coords[::-1]
    assert connected_groups(reversed_coords, 150.0) == [0, 1, 1]


def test_consumer_counts_are_symmetric_and_exclude_self() -> None:
    seps = [0.0, 50.0, 80.0, 5000.0]
    coords = SkyCoord([_offset(RA0, DEC0, s)[0] for s in seps] * u.deg,
                      [DEC0] * len(seps) * u.deg)
    assert consumer_counts(coords, 100.0) == [2, 2, 2, 0]
    assert consumer_counts(coords, 60.0) == [1, 2, 1, 0]


# ------------------------------------
# The gate census
# ------------------------------------

def test_census_reads_the_gate_the_engine_would_apply() -> None:
    coord = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
    reach = gate_reach_arcsec(120.0)
    rows = [
        _row(RA0, DEC0),                                  # the target
        _row(*_offset(RA0, DEC0, 20.0)),                  # gated neighbor
        _row(*_offset(RA0, DEC0, reach + 10.0)),          # past the reach
        _row(*_offset(RA0, DEC0, 25.0), kind='PSF'),      # point source
        _row(*_offset(RA0, DEC0, 30.0), flux_ujy=10.0),   # under the flux gate
        _row(*_offset(RA0, DEC0, 35.0), rchisq=1.0),      # not a misfit
    ]
    census = census_from_catalog(coord, pd.DataFrame(rows),
                                 cutout_arcsec=120.0)
    assert census['matched'] is True
    # 'gates' judges the target's own row at a non-self distance: the
    # question is whether ANOTHER field would seat it.
    assert census['gates'] is True
    # 'n_gated' counts the neighbors this field would shape-solve, so the
    # target's own row is excluded at its true distance of zero.
    assert census['n_gated'] == 1


def test_a_faint_target_does_not_gate() -> None:
    coord = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
    cat = pd.DataFrame([_row(RA0, DEC0, flux_ujy=10.0)])
    census = census_from_catalog(coord, cat, cutout_arcsec=120.0)
    assert census['matched'] is True
    assert census['gates'] is False


def test_an_unmatched_target_is_reported_not_gated() -> None:
    coord = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
    far = _row(*_offset(RA0, DEC0, recipe.TARGET_MATCH_AS + 5.0))
    census = census_from_catalog(coord, pd.DataFrame([far]),
                                 cutout_arcsec=120.0)
    assert census['matched'] is False
    assert census['gates'] is False

    empty = census_from_catalog(coord, pd.DataFrame(), cutout_arcsec=120.0)
    assert empty == {'matched': False, 'gates': False, 'n_gated': 0,
                     'entries': []}


# ------------------------------------
# The plan
# ------------------------------------

def _targets(seps, names=None):
    names = names or [f"t{i}" for i in range(len(seps))]
    return [{'name': name, 'label': name,
             'ra_deg': _offset(RA0, DEC0, sep)[0],
             'dec_deg': DEC0, 'dir': f"/tmp/{name}", 'priority': None}
            for name, sep in zip(names, seps)]


def _census(gates, consumers_unused=None, n_gated=1):
    return [{'matched': True, 'gates': g, 'n_gated': n_gated} for g in gates]


def test_only_gating_targets_with_consumers_join_the_harvest_pass() -> None:
    # t0 and t1 are close enough to read each other; t2 is isolated.
    targets = _targets([0.0, 60.0, 9000.0])
    plan = build_plan(targets, _census([True, False, True]),
                      cutout_arcsec=120.0)
    by_name = {r['name']: r for r in plan['targets']}

    # Gates and has a consumer -> harvest.
    assert by_name['t0']['pass'] == PASS_HARVEST
    # Has a consumer but does not gate -> harvests nothing about itself.
    assert by_name['t1']['pass'] == PASS_PARALLEL
    # Gates but nothing reads it -> the order buys nothing.
    assert by_name['t2']['pass'] == PASS_PARALLEL
    assert plan['n_harvest'] == 1
    assert plan['n_parallel'] == 2


def test_harvest_order_puts_the_most_read_record_first() -> None:
    # t1 sits between t0 and t2, so it is read by both.
    targets = _targets([0.0, 60.0, 120.0], names=['t0', 't1', 't2'])
    plan = build_plan(targets, _census([True, True, True]),
                      cutout_arcsec=120.0)
    harvest = sorted((r for r in plan['targets']
                      if r['pass'] == PASS_HARVEST),
                     key=lambda r: r['order'])
    assert [r['name'] for r in harvest] == ['t1', 't0', 't2']
    assert harvest[0]['n_consumers'] == 2


def test_explicit_priority_outranks_the_derived_order() -> None:
    targets = _targets([0.0, 60.0, 120.0], names=['t0', 't1', 't2'])
    targets[2]['priority'] = 10.0
    plan = build_plan(targets, _census([True, True, True]),
                      cutout_arcsec=120.0)
    harvest = sorted((r for r in plan['targets']
                      if r['pass'] == PASS_HARVEST),
                     key=lambda r: r['order'])
    assert harvest[0]['name'] == 't2'


def test_groups_are_disjoint_and_reported() -> None:
    # Two pairs, separated far past any link radius.
    targets = _targets([0.0, 60.0, 9000.0, 9060.0])
    plan = build_plan(targets, _census([True] * 4), cutout_arcsec=120.0)
    assert plan['n_harvest'] == 4
    assert plan['n_groups'] == 2
    assert plan['largest_group'] == 2

    groups = {}
    for row in plan['targets']:
        groups.setdefault(row['group'], []).append(row['name'])
    assert sorted(len(v) for v in groups.values()) == [2, 2]
    # Every group numbers its members from zero.
    for group in {r['group'] for r in plan['targets']}:
        orders = sorted(r['order'] for r in plan['targets']
                        if r['group'] == group)
        assert orders == list(range(len(orders)))


def test_a_larger_stamp_merges_groups() -> None:
    # Two pairs 350" apart. At the default stamp that gap exceeds the
    # link distance; a 400" stamp grows both radii past it.
    targets = _targets([0.0, 50.0, 400.0, 450.0])
    tight = build_plan(targets, _census([True] * 4), cutout_arcsec=120.0,
                       margin_arcsec=0.0)
    assert tight['n_harvest'] == 4
    assert tight['n_groups'] == 2

    wide = build_plan(targets, _census([True] * 4), cutout_arcsec=400.0,
                      margin_arcsec=0.0)
    assert wide['n_harvest'] == 4
    assert wide['n_groups'] == 1


def test_plan_records_its_own_radii_and_counts_unmatched() -> None:
    targets = _targets([0.0, 60.0])
    census = [{'matched': False, 'gates': False, 'n_gated': 0},
              {'matched': True, 'gates': True, 'n_gated': 3}]
    plan = build_plan(targets, census, cutout_arcsec=120.0)
    assert plan['cutout_arcsec'] == 120.0
    assert plan['scene_radius_arcsec'] == round(scene_radius_arcsec(120.0), 2)
    assert plan['gate_reach_arcsec'] == round(gate_reach_arcsec(120.0), 2)
    assert plan['link_radius_arcsec'] == round(link_radius_arcsec(120.0), 2)
    assert plan['n_unmatched'] == 1


def test_a_clean_plan_has_no_boundary_crossings() -> None:
    targets = _targets([0.0, 50.0, 9000.0, 9050.0])
    plan = build_plan(targets, _census([True] * 4), cutout_arcsec=120.0)
    assert plan['n_groups'] == 2
    assert plan['n_violations'] == 0


def _audit_rows(writer_group, reader_group, reader_pass=PASS_HARVEST):
    """Two fields 80" apart, the first harvesting a record 40" from it."""
    return [
        {'name': 'writer', 'ra_deg': RA0, 'dec_deg': DEC0,
         'pass': PASS_HARVEST, 'group': writer_group,
         'entries': [[_offset(RA0, DEC0, 40.0)[0], DEC0]]},
        {'name': 'reader', 'ra_deg': _offset(RA0, DEC0, 80.0)[0],
         'dec_deg': DEC0, 'pass': reader_pass, 'group': reader_group,
         'entries': []},
    ]


def test_the_audit_catches_a_record_crossing_a_group_boundary() -> None:
    violations = audit_groups(_audit_rows(0, 1), scene_arcsec=100.0)
    assert len(violations) == 1
    assert violations[0]['written_by'] == 'writer'
    assert violations[0]['read_by'] == 'reader'
    assert violations[0]['written_group'] != violations[0]['read_group']


def test_the_audit_passes_a_record_shared_inside_one_group() -> None:
    # Same record, same reach -- but one group, where a shared record is
    # the sequence working as intended rather than a race.
    assert audit_groups(_audit_rows(0, 0), scene_arcsec=100.0) == []


def test_the_audit_ignores_the_parallel_pass() -> None:
    # A parallel-pass field consumes the merged registry after the
    # freeze, so it sees every record regardless of who wrote it.
    rows = _audit_rows(0, 1, reader_pass=PASS_PARALLEL)
    assert audit_groups(rows, scene_arcsec=100.0) == []


def test_the_audit_ignores_a_record_out_of_reach() -> None:
    assert audit_groups(_audit_rows(0, 1), scene_arcsec=30.0) == []


def test_a_valid_margin_cannot_produce_a_crossing() -> None:
    # The link floor is write reach + read reach, so a record inside one
    # field's gate reach and another's scene cone puts the two fields
    # under the link distance -- i.e. in the same group by construction.
    for margin in (0.0, 60.0, DEFAULT_LINK_MARGIN_AS, 500.0):
        for seps in ([0.0, 50.0, 140.0], [0.0, 90.0, 180.0, 9000.0]):
            targets = _targets(seps)
            plan = build_plan(targets, _census([True] * len(seps)),
                              cutout_arcsec=120.0, margin_arcsec=margin)
            assert plan['n_violations'] == 0, (margin, seps)


def test_a_negative_margin_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        link_radius_arcsec(120.0, margin_arcsec=-1.0)
    with pytest.raises(ValueError, match="negative"):
        build_plan(_targets([0.0, 60.0]), _census([True, True]),
                   cutout_arcsec=120.0, margin_arcsec=-1.0)


def test_census_reports_the_positions_it_would_harvest() -> None:
    coord = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
    neighbor = _offset(RA0, DEC0, 20.0)
    cat = pd.DataFrame([_row(RA0, DEC0), _row(*neighbor)])
    census = census_from_catalog(coord, cat, cutout_arcsec=120.0)

    # The gated neighbor, plus the target's own position because it gates.
    assert len(census['entries']) == 2
    assert census['entries'][0] == pytest.approx([neighbor[0], neighbor[1]])
    assert census['entries'][1] == pytest.approx([RA0, DEC0])

    # A target that does not gate contributes no record of its own.
    faint = pd.DataFrame([_row(RA0, DEC0, flux_ujy=10.0), _row(*neighbor)])
    census = census_from_catalog(coord, faint, cutout_arcsec=120.0)
    assert len(census['entries']) == 1


def test_build_plan_rejects_a_misaligned_census() -> None:
    with pytest.raises(ValueError, match="2 targets but 1 census"):
        build_plan(_targets([0.0, 60.0]), _census([True]),
                   cutout_arcsec=120.0)


# ------------------------------------
# Target list and output
# ------------------------------------

def test_read_targets_accepts_the_shared_sample_catalog(tmp_path, capsys) -> None:
    path = tmp_path / "sample.csv"
    from sedphot.resolve import sanitize_label

    path.write_text("name,ra_deg,dec_deg,z_ref_kind,dir,label,priority\n"
                    "M 87,210.0,30.0,cluster,/data/a,m87_deep,\n"
                    "b,210.1,30.0,spec,,,5\n"
                    "M 87,210.0,30.0,cluster,/data/a,m87_deep,\n",
                    encoding="utf-8")
    targets = read_targets(path, out_root="/root")
    assert [t['name'] for t in targets] == ['M 87', 'b']
    # The fitter's columns ride along unrecognized. Reporting the split
    # is what keeps a typo in an optional column from taking a default
    # in silence, without this tool knowing the fitter's schema.
    line = capsys.readouterr().out
    assert "using name, ra_deg, dec_deg, dir, label, priority" in line
    assert "ignoring z_ref_kind" in line
    # An absolute dir is used as written; the relative case is covered
    # by test_dir_resolves_under_out_root_like_the_fitter_does.
    assert targets[0]['dir'] == '/data/a'
    # A named label wins; an unnamed one falls back to the sanitized name,
    # because the label is the product filename stem an existing tree
    # already committed to.
    assert targets[0]['label'] == 'm87_deep'
    assert targets[1]['label'] == sanitize_label('b')
    assert targets[1]['dir'] == "/root/b"     # blank dir falls back to name
    assert targets[1]['priority'] == 5.0
    assert targets[0]['priority'] is None


def test_read_targets_rejects_a_list_it_cannot_use(tmp_path) -> None:
    no_name = tmp_path / "no_name.csv"
    no_name.write_text("ra_deg,dec_deg\n210.0,30.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no name or target column"):
        read_targets(no_name)

    no_position = tmp_path / "no_pos.csv"
    no_position.write_text("name\na\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no ra_deg/dec_deg"):
        read_targets(no_position)

    # Without an out_root a relative dir is used as given, so a catalog
    # that names no directory still resolves against the caller's cwd.
    no_dir = tmp_path / "no_dir.csv"
    no_dir.write_text("name,ra_deg,dec_deg\na,210.0,30.0\n", encoding="utf-8")
    assert read_targets(no_dir)[0]['dir'] == 'a'


def test_write_plan_emits_json_and_an_ordered_csv(tmp_path) -> None:
    import csv as csv_module

    targets = _targets([0.0, 60.0, 9000.0])
    plan = build_plan(targets, _census([True, True, False]),
                      cutout_arcsec=120.0)
    json_path, csv_path = write_plan(plan, tmp_path / "plan.json")

    assert json.loads(json_path.read_text())['n_targets'] == 3
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv_module.DictReader(handle))
    # The harvest pass comes first, in run order; the parallel pass follows.
    assert [r['pass'] for r in rows] == [PASS_HARVEST, PASS_HARVEST,
                                         PASS_PARALLEL]
    assert [r['order'] for r in rows][:2] == ['0', '1']


def test_dir_resolves_under_out_root_like_the_fitter_does(tmp_path) -> None:
    # The seam this closes: sed_fitting resolves dir against a campaign's
    # data_root, so sedphot must resolve it against out_root or one shared
    # catalog cannot drive both tools.
    path = tmp_path / "sample.csv"
    path.write_text("name,ra_deg,dec_deg,dir\n"
                    "a,210.0,30.0,Galaxies/gal_a\n"
                    "b,210.1,30.0,/absolute/gal_b\n", encoding="utf-8")
    targets = read_targets(path, out_root="/cluster")
    assert targets[0]['dir'] == "/cluster/Galaxies/gal_a"
    # An absolute dir still wins, so an existing absolute catalog keeps working.
    assert targets[1]['dir'] == "/absolute/gal_b"
