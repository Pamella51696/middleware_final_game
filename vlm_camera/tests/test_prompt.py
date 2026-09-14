from vlm_camera.vlm.prompt import parse_scene_json, SCENE_PROMPT


def test_parse_plain_json():
    scene = parse_scene_json(
        '{"vehicles":["car"],"people":[],"hazards":[],"scene":"road","description":"ok","confidence":0.9}'
    )
    assert scene["vehicles"] == ["car"]
    assert scene["confidence"] == 0.9


def test_parse_fenced_and_junk():
    scene = parse_scene_json("Here you go:\n```json\n{\"vehicles\":[\"truck\"],\"scene\":\"st\"}\n```")
    assert scene["vehicles"] == ["truck"]
    assert scene["scene"] == "st"


def test_parse_invalid_returns_empty():
    scene = parse_scene_json("not json")
    assert scene["vehicles"] == []
    assert scene["scene"] == "unknown"


def test_prompt_asks_for_json_only():
    assert "JSON only" in SCENE_PROMPT
    assert "vehicles" in SCENE_PROMPT
