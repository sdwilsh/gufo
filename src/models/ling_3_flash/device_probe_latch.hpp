#pragma once

#include "src/cli/serve/text_model_runner.hpp"
#include "src/models/ling_3_flash/kernels/rocm/executor.hpp"

namespace gufo::models::ling_3_flash {

/// Applies a device-probe result to the process-wide device-suspect latch and
/// returns whether the device may still be used.
///
/// - kUsable: the probe completed, so the latch ends.
/// - kLost: the latch is set and the device is unusable.
/// - kPending: the probe did not finish (a timeout, or an idle poll that is
///   still running). It is inconclusive: the latch stays as it is, and the
///   device is not declared lost.
///
/// The idle-loop poll (text_generation_scheduler) treats only kLost as an
/// event. It does not call this: its loss path ends in MarkDeviceLost, which
/// retires every session, so the latch adds nothing there, and a usable idle
/// poll must not clear a latch that a failed work unit set before the
/// scheduler classified it.
inline bool ApplyDeviceProbe(
    server::TextModelRunner::DeviceProbeStatus status) {
  using Status = server::TextModelRunner::DeviceProbeStatus;
  switch (status) {
    case Status::kUsable:
      rocm::Executor::ClearDeviceSuspect();
      return true;
    case Status::kLost:
      rocm::Executor::NoteDeviceError();
      return false;
    case Status::kPending:
      return true;
  }
  return true;
}

}  // namespace gufo::models::ling_3_flash
