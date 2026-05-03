#pragma once

// ============================================================================
// FormParams + TauMode — параметры мультиканального генератора FormSignal
//
// ЧТО:    POD-структура с 18+ полями (sampling / signal / noise / per-channel
//         delay / freq range) для FormSignalGenerator(ROCm). Включает:
//         - enum TauMode (FIXED / LINEAR / RANDOM) — режим per-channel задержки;
//         - GetTauMode() — авто-детект режима по заполненным полям;
//         - GetDuration() / GetDt() — производные от fs/points;
//         - ParseFromString("f0=1e6,a=1.0,an=0.1,tau=0.001") — парсер из DSL,
//           поддерживает 17 ключей, кидает std::invalid_argument при ошибке.
//
// ЗАЧЕМ:  FormSignal — самый сложный сигнал в модуле (chirp + Gaussian noise
//         + per-channel tau с тремя режимами). Передавать 18 параметров через
//         ctor нечитаемо. Структура с дефолтами + DSL-парсер позволяют:
//         - задавать только нужные поля (остальное — дефолты);
//         - конфигурировать из Python через строку;
//         - сравнивать наборы параметров для кеширования (operator== не
//           нужен — кеш живёт через SignalService).
//
// ПОЧЕМУ: - POD + helpers (не методы full-fledged класса) — данные открыты
//           для прямого доступа из kernel-launcher'а, не плодим getter'ы.
//         - GetTauMode() возвращает enum (не bool flags) → kernel получает
//           одно число tau_mode ∈ {0,1,2}, ветвится без switch.
//         - ParseFromString static — fail-fast (бросает на неизвестном ключе),
//           caller сразу видит опечатку.
//         - norm = 1/√2 по умолчанию → нормализация мощности сигнала I/Q.
//
// Использование:
//   // Из Python через DSL:
//   auto params = FormParams::ParseFromString("f0=1e6,a=1.0,fdev=2e6,an=0.1");
//
//   // Прямо в C++:
//   FormParams p;
//   p.f0 = 1e6; p.amplitude = 1.0; p.tau_step = 1e-6;  // LINEAR mode
//   assert(p.GetTauMode() == TauMode::LINEAR);
//
// История:
//   - Создан:  2026-02-17
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#include <cstdint>
#include <cmath>
#include <string>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace signal_gen {

// ════════════════════════════════════════════════════════════════════════════
// Режим задержки per-channel
// ════════════════════════════════════════════════════════════════════════════

enum class TauMode {
  FIXED,    ///< tau = tau_base (одинаковая для всех)
  LINEAR,   ///< tau = tau_base + ID * tau_step
  RANDOM    ///< tau = tau_min + uniform[0,1) * (tau_max - tau_min)
};

// ════════════════════════════════════════════════════════════════════════════
// FormParams
// ════════════════════════════════════════════════════════════════════════════

/**
 * @struct FormParams
 * @brief Параметры мультиканального FormSignal-генератора (chirp + noise + tau).
 *
 * @note Aggregate-инициализация работает (все поля public, без user-defined ctor).
 * @note ParseFromString — единственная точка валидации синтаксиса DSL-строки.
 * @see TauMode
 * @see SignalRequest
 */
struct FormParams {
  // --- Sampling ---
  double fs = 12e6;             ///< Частота дискретизации (Hz)
  uint32_t antennas = 1;        ///< Количество антенн/каналов
  uint32_t points = 4096;       ///< Отсчётов на антенну

  // --- Signal ---
  double f0 = 0.0;              ///< Центральная частота (Hz)
  double amplitude = 1.0;       ///< Амплитуда сигнала (a)
  double phase = 0.0;           ///< Начальная фаза phi (rad)
  double fdev = 0.0;            ///< Девиация частоты (chirp), 0 = CW
  double norm = 0.7071067811865476;  ///< Нормировка, 1/sqrt(2) по умолчанию

  // --- Noise ---
  double noise_amplitude = 0.0; ///< Амплитуда шума (an), 0 = без шума
  uint32_t noise_seed = 0;      ///< Seed для шума, 0 = random

  // --- Delay per-channel: линейный ---
  double tau_base = 0.0;        ///< Базовая задержка (с)
  double tau_step = 0.0;        ///< Шаг задержки на канал (с)

  // --- Delay per-channel: случайный ---
  double tau_min = 0.0;         ///< Мин задержка (с) для RANDOM
  double tau_max = 0.0;         ///< Макс задержка (с) для RANDOM
  uint32_t tau_seed = 12345;    ///< Seed для случайной задержки

  // --- Frequency range (опционально) ---
  double freq_min = 0.0;        ///< Мин частота для multi-beam
  double freq_max = 0.0;        ///< Макс частота для multi-beam

  // ══════════════════════════════════════════════════════════════════════
  // Helpers
  // ══════════════════════════════════════════════════════════════════════

  /// Определить режим задержки по заполненным полям
  TauMode GetTauMode() const {
    if (tau_step != 0.0)
      return TauMode::LINEAR;
    if (tau_min != tau_max && (tau_min != 0.0 || tau_max != 0.0))
      return TauMode::RANDOM;
    return TauMode::FIXED;
  }

  /// Длительность сигнала (с)
  double GetDuration() const {
    return static_cast<double>(points) / fs;
  }

  /// Шаг дискретизации (с)
  double GetDt() const {
    return 1.0 / fs;
  }

  // ══════════════════════════════════════════════════════════════════════
  // Парсер из строки
  // ══════════════════════════════════════════════════════════════════════

  /**
   * @brief Парсинг параметров из строки
   * @param str Строка вида "f0=1e6,a=1.0,an=0.1,tau=0.001"
   * @return FormParams с заполненными полями
   * @throws std::invalid_argument при ошибке парсинга
   *
   * Поддерживаемые ключи:
   * fs, f0, a, an, phi, fdev, norm, tau, tau_step, tau_min, tau_max,
   * tau_seed, noise_seed, antennas, points, freq_min, freq_max
   *   @test_check parses "f0=1e6,a=1.0" → result.f0==1e6 && result.amplitude==1.0
   *   @test_check throws std::invalid_argument on unknown key, missing '=', non-numeric value, out_of_range
   */
  static FormParams ParseFromString(const std::string& str) {
    FormParams p;
    std::string s = str;

    // Убрать пробелы
    s.erase(std::remove_if(s.begin(), s.end(),
        [](unsigned char c) { return std::isspace(c); }), s.end());

    if (s.empty()) return p;

    std::istringstream stream(s);
    std::string token;

    while (std::getline(stream, token, ',')) {
      auto eq_pos = token.find('=');
      if (eq_pos == std::string::npos) {
        throw std::invalid_argument(
            "FormParams::ParseFromString: invalid token '" + token + "'");
      }

      std::string key = token.substr(0, eq_pos);
      std::string val = token.substr(eq_pos + 1);

      // Привести key к нижнему регистру
      std::transform(key.begin(), key.end(), key.begin(),
          [](unsigned char c) { return std::tolower(c); });

      try {
        if      (key == "fs")         p.fs = std::stod(val);
        else if (key == "f0")         p.f0 = std::stod(val);
        else if (key == "a")          p.amplitude = std::stod(val);
        else if (key == "an")         p.noise_amplitude = std::stod(val);
        else if (key == "phi")        p.phase = std::stod(val);
        else if (key == "fdev")       p.fdev = std::stod(val);
        else if (key == "norm")       p.norm = std::stod(val);
        else if (key == "tau")        p.tau_base = std::stod(val);
        else if (key == "tau_step")   p.tau_step = std::stod(val);
        else if (key == "tau_min")    p.tau_min = std::stod(val);
        else if (key == "tau_max")    p.tau_max = std::stod(val);
        else if (key == "tau_seed")   p.tau_seed = static_cast<uint32_t>(std::stoul(val));
        else if (key == "noise_seed") p.noise_seed = static_cast<uint32_t>(std::stoul(val));
        else if (key == "antennas")   p.antennas = static_cast<uint32_t>(std::stoul(val));
        else if (key == "points")     p.points = static_cast<uint32_t>(std::stoul(val));
        else if (key == "freq_min")   p.freq_min = std::stod(val);
        else if (key == "freq_max")   p.freq_max = std::stod(val);
        else {
          throw std::invalid_argument(
              "FormParams::ParseFromString: unknown key '" + key + "'");
        }
      } catch (const std::invalid_argument&) {
        throw std::invalid_argument(
            "FormParams::ParseFromString: cannot parse value for '" + key + "'");
      } catch (const std::out_of_range&) {
        throw std::invalid_argument(
            "FormParams::ParseFromString: value out of range for '" + key + "'");
      }
    }

    return p;
  }
};

} // namespace signal_gen
