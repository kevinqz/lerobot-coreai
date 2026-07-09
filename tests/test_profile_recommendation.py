# test_profile_recommendation.py — profile recommendation heuristics (v0.9.1).

from lerobot_coreai.profile_recommendation import recommend_profile


def test_so100_from_robot_type():
    r = recommend_profile(robot_type="so100")
    assert r.recommended_profile == "so100-sim-default"
    assert r.confidence == "high"


def test_so101_from_robot_type():
    r = recommend_profile(robot_type="so101")
    assert r.recommended_profile == "so101-sim-default"


def test_pusht_from_shape():
    r = recommend_profile(dominant_shape=[2])
    assert r.recommended_profile == "pusht-sim-default"


def test_pusht_from_env_id():
    r = recommend_profile(env_id="PushT-v0")
    assert r.recommended_profile == "pusht-sim-default"


def test_generic_7dof_from_shape():
    r = recommend_profile(dominant_shape=[7])
    assert r.recommended_profile == "generic-7dof-sim-default"


def test_so100_from_shape():
    r = recommend_profile(dominant_shape=[16, 7])
    assert r.recommended_profile == "so100-sim-default"


def test_unknown_falls_back_to_default():
    r = recommend_profile()
    assert r.recommended_profile == "default-sim-safe"
    assert r.confidence == "low"


def test_always_warns_no_physical_safety():
    for r in (recommend_profile(robot_type="so100"), recommend_profile(),
              recommend_profile(dominant_shape=[2])):
        assert any("physical safety" in w for w in r.warnings)


def test_robot_type_beats_shape():
    # Robot type is the strongest signal, overriding a conflicting shape.
    r = recommend_profile(robot_type="so100", dominant_shape=[2])
    assert r.recommended_profile == "so100-sim-default"
