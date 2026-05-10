# Архитектурные паттерны репо `signal_generators`

> **Источник истины:** `signal_generators/.rag/_RAG.md` (теги `#pattern:Type:Class`, auto-inferred RAG_CLAUDE_C4 от 9.05).
> Brief'ы — из `key_classes:` того же манифеста (fallback из `rag_dsp.symbols`).
>
> Используется как источник для `dataset_v4` (collect_doc_deep подхватит Doc/Patterns.md).
> Alex: проверить + добавить руками то что не размечено в `_RAG.md tags:`.

## Facade

> Тонкий публичный API над набором операций. Стабильный → Python-биндинги не ломаются.


- **`signal_gen::FormSignalGenerator`** — `signal_generators/include/signal_generators/generators/form_signal_generator.hpp:67`
  - GPU-генератор комплексных сигналов по формуле getX
- **`signal_gen::ScriptGenerator`** — `signal_generators/include/signal_generators/generators/script_generator.hpp:83`
  - Compiles text DSL -> OpenCL kernel and executes on GPU

## Strategy

> Семейство взаимозаменяемых алгоритмов за общим интерфейсом (`IPipelineStep`).


- **`signal_gen::ISignalGenerator`** — `signal_generators/include/signal_generators/i_signal_generator.hpp:31`
  - Абстрактный интерфейс генератора сигналов * Реализации: - CwGenerator — синусоида (CW) - LfmGenerator — линейная ЧМ (chirp) - NoiseGenerat

## Factory

> Создание объектов по типу/конфигу без раскрытия конкретных классов.


- **`signal_gen::SignalGeneratorFactory`** — `signal_generators/include/signal_generators/signal_generator_factory.hpp:31`
  - Создаёт генераторы по типу сигнала


## См. также

- `signal_generators/.rag/arch/C2_container.md`
- `signal_generators/.rag/arch/C3_component.md`
- `signal_generators/.rag/arch/C4_code.md`
- `signal_generators/Doc/Architecture.md`
- `MemoryBank/.architecture/DSP-GPU_Design_C4_Full.md`

---

*Сгенерировано из `_RAG.md` тегов. Alex редактирует руками + коммитит.*
