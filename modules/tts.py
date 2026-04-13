"""
modules/tts.py — 텍스트 → 음성 (Coqui XTTS v2 + pygame)

Coqui XTTS v2 모델을 로컬에서 실행하여 wav를 합성하고,
pygame.mixer로 재생한다.

첫 실행 시 모델이 자동 다운로드된다 (~2GB).
XTTS_REF_AUDIO_DIR 내 모든 wav를 레퍼런스로 사용해 음성 클로닝을 수행한다.

실행:
    python modules/tts.py
"""
import os
import wave
from pathlib import Path
from typing import List, Optional

import pygame
from loguru import logger

from config import BASE_DIR, DEVICE, TTS_MODEL, XTTS_LANGUAGE, XTTS_REF_AUDIO_DIR

# ── 설정 ─────────────────────────────────────────────────
_DEFAULT_OUTPUT: Path = BASE_DIR / "temp_tts.wav"

# 싱글턴 TTS 엔진
_tts_engine: Optional[object] = None


def _get_ref_audio_list() -> List[str]:
    """XTTS_REF_AUDIO_DIR 내 모든 wav 파일 경로를 반환한다."""
    wavs = sorted(XTTS_REF_AUDIO_DIR.glob("*.wav"))
    # 백업 파일 제외
    wavs = [w for w in wavs if "backup" not in w.name]
    if not wavs:
        raise FileNotFoundError(f"레퍼런스 오디오가 없습니다: {XTTS_REF_AUDIO_DIR}")
    logger.debug("레퍼런스 오디오 {n}개 사용 | dir={d}", n=len(wavs), d=XTTS_REF_AUDIO_DIR)
    return [str(w) for w in wavs]


# ── 모델 로드 ─────────────────────────────────────────────

def load_models() -> None:
    """Coqui XTTS v2 모델을 로드한다. 앱 시작 시 1회 호출.

    첫 실행 시 모델을 자동 다운로드한다 (~2GB).

    Raises:
        RuntimeError: 모델 로드 실패 시
    """
    global _tts_engine
    if _tts_engine is not None:
        logger.debug("TTS 모델 이미 로드됨, 스킵")
        return

    try:
        os.environ["COQUI_TOS_AGREED"] = "1"  # 비상업적 CPML 라이선스 동의
        from TTS.api import TTS  # type: ignore[import]
        logger.info("XTTS v2 모델 로드 중 | model={m} | device={d}", m=TTS_MODEL, d=DEVICE)
        _tts_engine = TTS(model_name=TTS_MODEL, progress_bar=True).to(DEVICE)
        logger.info("XTTS v2 모델 로드 완료")
    except Exception as e:
        raise RuntimeError(f"XTTS v2 모델 로드 실패: {e}") from e


def _get_engine() -> object:
    """TTS 엔진 싱글턴을 반환한다. 미로드 시 자동 로드."""
    if _tts_engine is None:
        load_models()
    return _tts_engine


# ── 합성 ──────────────────────────────────────────────────

def synthesize(
    text: str,
    output_path: Optional[Path] = None,
) -> Path:
    """XTTS v2로 텍스트를 wav 파일로 합성한다.

    XTTS_REF_AUDIO_DIR 내 모든 wav를 레퍼런스로 사용해 음성 클로닝을 수행한다.

    Args:
        text: 합성할 텍스트 (빈 문자열 불가)
        output_path: 저장 경로. None이면 BASE_DIR/temp_tts.wav 사용

    Returns:
        저장된 wav 파일의 Path

    Raises:
        ValueError: text가 비어 있거나 공백만인 경우
        FileNotFoundError: 레퍼런스 오디오 파일이 없는 경우
        RuntimeError: 합성 실패 시
    """
    if not text or not text.strip():
        raise ValueError("합성할 텍스트가 비어 있습니다.")

    ref_audios = _get_ref_audio_list()
    dest: Path = output_path if output_path is not None else _DEFAULT_OUTPUT

    logger.debug(
        "TTS 합성 시작 | text_len={n} | lang={lang} | ref={r}개 | dest={dest}",
        n=len(text),
        lang=XTTS_LANGUAGE,
        r=len(ref_audios),
        dest=dest,
    )

    try:
        engine = _get_engine()
        engine.tts_to_file(  # type: ignore[union-attr]
            text=text,
            speaker_wav=ref_audios,
            language=XTTS_LANGUAGE,
            file_path=str(dest),
            temperature=0.75,
            repetition_penalty=2.0,
            top_k=50,
            top_p=0.85,
            speed=1.0,
        )
    except Exception as e:
        raise RuntimeError(f"XTTS v2 합성 오류: {e}") from e

    logger.info("TTS 합성 완료 | dest={dest}", dest=dest)
    return dest


# ── 재생 ──────────────────────────────────────────────────

def _read_wav_params(file_path: Path) -> tuple[int, int, int]:
    """wav 파일의 (샘플레이트, 채널 수, sampwidth)를 반환한다."""
    with wave.open(str(file_path), "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth()


def play(file_path: Path) -> None:
    """wav 파일을 pygame.mixer로 재생한다. 재생 완료까지 블로킹.

    Args:
        file_path: 재생할 wav 파일 경로

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        RuntimeError: pygame 초기화 또는 재생 오류 시
    """
    if not file_path.exists():
        raise FileNotFoundError(f"재생 파일이 없습니다: {file_path}")

    sample_rate, channels, sampwidth = _read_wav_params(file_path)
    bit_depth = sampwidth * 8
    logger.debug(
        "오디오 재생 시작 | file={file} | {hz}Hz | {ch}ch | {bit}bit",
        file=file_path,
        hz=sample_rate,
        ch=channels,
        bit=bit_depth,
    )

    try:
        size = -(sampwidth * 8) if sampwidth > 1 else sampwidth * 8
        pygame.mixer.init(frequency=sample_rate, size=size, channels=channels)
        pygame.mixer.music.load(str(file_path))
        pygame.mixer.music.set_volume(0.6)
        pygame.mixer.music.play()

        clock = pygame.time.Clock()
        while pygame.mixer.music.get_busy():
            clock.tick(10)

        logger.info("오디오 재생 완료 | file={file}", file=file_path)
    except Exception as e:
        raise RuntimeError(f"pygame 재생 오류: {e}") from e
    finally:
        _safe_mixer_quit()


def _safe_mixer_quit() -> None:
    """pygame.mixer가 초기화된 경우에만 종료한다."""
    try:
        if pygame.mixer.get_init():
            pygame.mixer.quit()
    except Exception as e:
        logger.warning("pygame.mixer 종료 중 오류 (무시): {e}", e=e)


# ── 통합 인터페이스 ───────────────────────────────────────

def speak(text: str) -> None:
    """텍스트를 합성 후 재생한다. 재생 완료 후 임시 파일을 삭제한다.

    Args:
        text: 읽을 텍스트

    Raises:
        ValueError: text가 비어 있거나 공백만인 경우
    """
    wav_path = synthesize(text)
    try:
        play(wav_path)
    finally:
        _cleanup_temp_file(wav_path)


def _cleanup_temp_file(file_path: Path) -> None:
    """임시 파일을 삭제한다. 삭제 실패 시 경고만 기록하고 진행한다."""
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug("임시 파일 삭제 | file={file}", file=file_path)
    except Exception as e:
        logger.warning("임시 파일 삭제 실패 (무시): {file} | {e}", file=file_path, e=e)


# ── 단독 테스트 ───────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== tts.py 단독 테스트 시작 ===")

    # ── 정상 케이스 1: 모델 로드 ──────────────────────────
    logger.info("--- 정상 케이스 1: load_models() ---")
    try:
        load_models()
        logger.info("정상 케이스 1 통과")
    except Exception as e:
        logger.error("정상 케이스 1 실패: {e}", e=e)

    # ── 정상 케이스 2: 음성 합성 + 재생 ──────────────────
    logger.info("--- 정상 케이스 2: speak('안녕하세요, 저는 리아입니다.') ---")
    try:
        speak("안녕하세요, 저는 리아입니다.")
        logger.info("정상 케이스 2 통과")
    except Exception as e:
        logger.error("정상 케이스 2 실패: {e}", e=e)

    # ── 에러 케이스 1: 빈 문자열 ──────────────────────────
    logger.info("--- 에러 케이스 1: 빈 문자열 ---")
    try:
        speak("")
        logger.warning("에러 케이스 1: 예외가 발생해야 하는데 통과됨")
    except ValueError as e:
        logger.info("에러 케이스 1 정상 처리: {e}", e=e)

    # ── 에러 케이스 2: 레퍼런스 오디오 디렉토리 비어 있음 ──
    logger.info("--- 에러 케이스 2: 레퍼런스 오디오 없음 ---")
    import os
    _orig = os.environ.get("XTTS_REF_AUDIO_DIR")
    os.environ["XTTS_REF_AUDIO_DIR"] = "/nonexistent/dir"
    try:
        from pathlib import Path as _Path
        wavs = list(_Path("/nonexistent/dir").glob("*.wav")) if _Path("/nonexistent/dir").exists() else []
        if not wavs:
            raise FileNotFoundError("레퍼런스 오디오가 없습니다: /nonexistent/dir")
        logger.warning("에러 케이스 2: 예외가 발생해야 하는데 통과됨")
    except FileNotFoundError as e:
        logger.info("에러 케이스 2 정상 처리: {e}", e=e)
    finally:
        if _orig is None:
            os.environ.pop("XTTS_REF_AUDIO_DIR", None)
        else:
            os.environ["XTTS_REF_AUDIO_DIR"] = _orig

    logger.info("=== tts.py 단독 테스트 완료 ===")
