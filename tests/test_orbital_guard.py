def test_root_and_preset_simulation_endpoints():
    client = TestClient(app)

    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert root_resp.json()["service"] == "OrbitalGuard-BT"

    glare_resp = client.post("/simulation/run-preset/glare_recovery")
    assert glare_resp.status_code == 200
    assert glare_resp.json()["final_mode"] == "COMPLETED"

    blackout_resp = client.post("/simulation/run-preset/blackout_safe_hold")
    assert blackout_resp.status_code == 200
    assert blackout_resp.json()["final_mode"] == "SAFE_HOLD"