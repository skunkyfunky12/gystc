from data.regions import REGIONS, COMMUNITY_TO_REGION


def test_twelve_regions():
    assert len(REGIONS) == 12


def test_each_region_has_required_fields():
    for r in REGIONS:
        assert "name" in r
        assert "color" in r and len(r["color"]) == 3
        assert "position" in r and len(r["position"]) == 3


def test_community_mapping_covers_all():
    for cid in range(77):
        assert cid in COMMUNITY_TO_REGION
        assert 0 <= COMMUNITY_TO_REGION[cid] < 12
