// Exercise the real SDK callback/parser without loading a headset service.
#include <cassert>
#include <cstring>
#include <pybind11/embed.h>
#include "../../../scripts/pico4/sdk/bindings/py_bindings.cpp"

extern "C" int PXREAInit(void*, pfPXREAClientCallback, unsigned) { return 0; }
extern "C" int PXREADeinit() { return 0; }

void emit(const json& value) {
    PXREADevStateJson message{};
    const auto text = json{{"value", value.dump()}}.dump();
    assert(text.size() < sizeof(message.stateJson));
    std::strcpy(message.stateJson, text.c_str());
    OnPXREAClientCallback(nullptr, PXREADeviceStateJson, 0, &message);
}

int main() {
    pybind11::scoped_interpreter interpreter{};
    json joints = json::array();
    for (int i = 0; i < 26; ++i) joints.push_back({{"p", "0,1,2,0,0,0,1"}});
    json hand{{"scale", 1.0}, {"isActive", 1}, {"HandJointLocations", joints}};
    json packet{{"timeStampNs", 42}, {"Head", {{"pose", "0,1.5,0,0,0,0,1"}}},
                {"Hand", {{"leftHand", hand}, {"rightHand", hand}}}};
    emit(packet);
    assert(HandInput.active[0] && HandInput.active[1]);
    assert(HandInput.headValid && HandInput.stamp == 42);
    assert(HandInput.joints[0][1][1] == 1.0);
    auto snapshot = getHandInputSample();
    assert(snapshot["age_s"].cast<double>() < 0.25);
    assert(snapshot["left"].cast<pybind11::dict>()["active"].cast<bool>());

    // Missing hand data cannot inherit validity from the previous packet.
    auto missing = packet;
    missing["Hand"].erase("leftHand");
    emit(missing);
    assert(!HandInput.active[0] && HandInput.active[1]);
    auto inactive = packet;
    inactive["Hand"]["rightHand"]["isActive"] = 0;
    emit(inactive);
    assert(HandInput.active[0] && !HandInput.active[1]);
    auto shortHand = packet;
    shortHand["Hand"]["leftHand"]["HandJointLocations"] = json::array();
    emit(shortHand);
    assert(!HandInput.active[0]);

    // Malformed joint parsing must invalidate the whole packet and not escape.
    auto malformed = packet;
    malformed["Hand"]["leftHand"]["HandJointLocations"][1]["p"] = "invalid";
    emit(malformed);
    assert(!HandInput.active[0] && !HandInput.active[1]);
    for (const auto* bad : {"0,1,2,1", "0,1,2,0,0,0,1,", "0,1,2,0,0,0,1junk"}) {
        malformed["Hand"]["leftHand"]["HandJointLocations"][1]["p"] = bad;
        emit(malformed);
        assert(!HandInput.active[0] && !HandInput.active[1]);
    }
    emit(packet);
    auto headOnly = packet;
    headOnly.erase("Hand");
    emit(headOnly);
    assert(!HandInput.active[0] && !HandInput.active[1]);
    auto noHead = packet;
    noHead.erase("Head");
    emit(noHead);
    assert(!HandInput.headValid);
    emit(packet);
    OnPXREAClientCallback(nullptr, PXREAServerDisconnect, 0, nullptr);
    assert(!HandInput.active[0] && !HandInput.active[1]);
    emit(packet);
    init();  // Stubbed init resets a prior cached sample.
    assert(!HandInput.active[0] && HandInput.stamp == 0);
}
